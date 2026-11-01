"""Single-scan diagnostic: legacy visual object matching + explicit directed metrics."""
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw
from common import sha256, write_json
from dashscope_client import parse_json
from local_matching_client import LocalMatchingClient as DashScopeClient

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]


def load_functions(path, names=None):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    for n in nodes:
        n.decorator_list = []
    namespace = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace



def matching_candidates(data, candidate_ids):
    from matching_compat import candidate_record
    ids, scores, _ = candidate_record(data, candidate_ids)
    return ids, scores


def comparable_messages(messages):
    return [{'role':m['role'], 'content':[
        {'type':'image', 'sha256':sha256(c['image'])} if c['type']=='image' else c
        for c in m['content']]} for m in messages]


def metrics(gt, pred, mapping, k):
    gids = {str(n['id']) for n in gt['nodes']}
    pids = {str(n['id']) for n in pred['nodes']}
    assert set(mapping) == gids
    matched = [p for p in mapping.values() if p != 'none']
    assert len(matched) == len(set(matched)) and set(matched) <= pids
    pairs = {(str(e['src']), str(e['tgt'])): e['c_edge'][:k] for e in pred['edges']}
    assert len(pairs) == len(pred['edges'])
    counts, hits = Counter(), Counter()
    details = []
    both = hit = predicted_pair = 0
    for e in gt['edges']:
        s, t, p = str(e['src']), str(e['tgt']), e['relationship']
        assert s in gids and t in gids and isinstance(p, str) and p
        ps, pt = mapping[s], mapping[t]
        mapped = ps != 'none' and pt != 'none'
        proposed = pairs.get((ps, pt), []) if mapped else []
        ok = p in proposed
        both += int(mapped)
        predicted_pair += int(bool(proposed))
        hit += int(ok)
        counts[p] += 1
        hits[p] += int(ok)
        details.append({'gt_src': s, 'gt_tgt': t, 'gt_predicate': p,
                        'pred_src': ps, 'pred_tgt': pt, 'proposals': proposed,
                        'both_endpoints_matched': mapped, 'hit': ok})
    total = len(details)
    return {'k_per_directed_pair': k, 'object_hits': len(matched), 'gt_objects': len(gids),
            'object_recall': len(matched)/len(gids), 'gt_relation_rows': total,
            'both_endpoints_matched_rows': both, 'mapped_pair_with_prediction_rows': predicted_pair,
            'relationship_hits': hit, 'relationship_recall_all_gt': hit/total if total else None,
            'predicate_recall_given_matched_endpoints': hit/both if both else None,
            'predicate_recall_given_existing_predicted_pair': hit/predicted_pair if predicted_pair else None,
            'relationship_mrecall_all_gt_present_predicates':
                sum(hits[p]/n for p, n in counts.items())/len(counts) if counts else None,
            'per_predicate': {p: {'hits': hits[p], 'total': n} for p, n in sorted(counts.items())},
            'details': details}


def visibility(gt, bboxes, retained):
    frames = {Path(p).name.removesuffix('.color.jpg') for p in retained}
    rows = []
    for n in gt['nodes']:
        info = bboxes[str(n['id'])]
        valid = {f for f, box in zip(info['seen_frames'], info['bounding_boxes'])
                 if (box[2]-box[0])*(box[3]-box[1]) > 1000}
        rows.append({'gt_id': str(n['id']), 'label': n['label'],
                     'eligible_bbox_frames': len(valid),
                     'eligible_bbox_frames_sent_to_generator': sorted(valid & frames)})
    return {'min_bbox_area_exclusive': 1000, 'unique_generator_frames': len(frames),
            'objects_visible_in_generator_frames': sum(bool(x['eligible_bbox_frames_sent_to_generator']) for x in rows),
            'gt_objects': len(rows), 'objects': rows,
            'interpretation': 'Bounding-box coverage diagnostic; not object recall or proof that filtering caused a miss.'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=EXPERIMENT/'runs/single-scan-112914')
    parser.add_argument('--output-name', default='evaluation_qwen37_seed0')
    parser.add_argument('--reuse-requests', type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    out = run/args.output_name
    out.mkdir(exist_ok=False)
    (out/'images').mkdir()
    manifest = json.loads((run/'manifest.json').read_text())
    cfg = manifest['config']
    scan = manifest['arguments']['scan']
    gt_path = Path(cfg['data_root'])/scan/f'{scan}_scan_sg.json'
    bbox_path = gt_path.with_name(f'{scan}_gt_bboxes.json')
    pred_path = run/'graph_final.json'
    gt, pred = json.loads(gt_path.read_text()), json.loads(pred_path.read_text())
    bboxes = json.loads(bbox_path.read_text())[scan]
    source = EXPERIMENT/'legacy_matching/eva_qwen8b.py'
    legacy_eval = EXPERIMENT/'legacy_matching/eva_final_no_vlm.py'
    api_cfg = dict(cfg['api'], max_tokens=2048)
    client = DashScopeClient(api_cfg)
    conflict_client = DashScopeClient(dict(cfg['api'],max_tokens=1024))
    if args.reuse_requests:
        prior = json.loads((args.reuse_requests/'manifest.json').read_text())
        assert prior['matching_api'] == client.config
    from matching_compat import PARSER_SHA256
    inputs = [gt_path, bbox_path, pred_path, run/'manifest.json', source, legacy_eval, Path(__file__), Path(__file__).with_name('matching_compat.py')]
    for p in [source, legacy_eval, Path(__file__), Path(__file__).with_name('matching_compat.py')]:
        (out/p.name).write_bytes(p.read_bytes())
    write_json(out/'manifest.json', {'scope': 'diagnostic_only', 'scan': scan, 'generation_run': str(run),
        'matching_api': client.config, 'conflict_api': conflict_client.config, 'seed': None, 'sampling': 'Original unseeded random.sample; actual image hashes recorded', 'max_images_per_gt': 5, 'min_bbox_area_exclusive': 1000,
        'candidate_parser_sha256': PARSER_SHA256,
        'format_policy': 'Original parser preserves duplicates/scalar conversion/padding; bad evidence abstains per GT; unknown IDs cannot become matches',
        'matching': 'Original candidate prompt, single conflict call, three-pass resolver and set iteration; Shared local Qwen3-VL-32B/vLLM client; direct gt2pred for primary scoring.',
        'relation_protocol': 'Directed, exact predicate strings, top K per pair; all GT relation rows remain in denominator; predicate judge disabled.',
        'object_protocol': 'One-to-one visual instance correspondence; category disagreement recorded separately, as legacy object hits do not require identical labels.',
        'reuse_requests': str(args.reuse_requests) if args.reuse_requests else None,
        'inputs_sha256': {str(p): sha256(p) for p in inputs}})
    helpers = load_functions(source, {'prepare_images_with_selection', 'build_prompt_for_matching',
                                      'build_prompt_for_pred_vs_gt', 'pred_vs_gt'})
    helpers.update(SCAN_DIR=cfg['data_root'], TEMP_IMG_DIR=str(out/'images'), MIN_BBOX_AREA=1000,
                   delete_temporary_images=lambda _: None)
    request_count = 0
    counter_lock = threading.Lock()

    def request(messages, name, active_client=None):
        nonlocal request_count
        with counter_lock:
            request_count += 1
            ordinal = request_count
        prefix = out/f'{ordinal:03d}_{name}'
        write_json(str(prefix)+'.messages.json', messages)
        paths = [c['image'] for m in messages for c in m['content'] if c['type']=='image']
        write_json(str(prefix)+'.images.json', {p: sha256(p) for p in paths})
        cache = args.reuse_requests / (prefix.name + '.response.json') if args.reuse_requests else None
        if cache and cache.exists():
            cached_messages = json.loads(cache.with_name(prefix.name+'.messages.json').read_text())
            assert comparable_messages(cached_messages) == comparable_messages(messages)
            response = json.loads(cache.read_text())
            expected = hashlib.sha256(json.dumps((active_client or client).payload(messages),ensure_ascii=False).encode()).hexdigest()
            if response.get('request_sha256') != expected:raise ValueError('Cached full request differs; reuse refused')
            write_json(str(prefix)+'.reused.json', {'source':str(cache.resolve()), 'sha256':sha256(cache)})
        else:
            response = (active_client or client).complete(messages)
        write_json(str(prefix)+'.response.json', response)
        try:
            if not isinstance(response.get('text'),str):raise ValueError('Non-text model reply')
            data = json.loads(response['text'])
            if not isinstance(data,dict):raise ValueError('Expected a JSON object')
            return data
        except (ValueError,TypeError,KeyError) as error:
            write_json(str(prefix)+'.parse_warning.json', {'type':type(error).__name__,
                       'policy':'Legacy per-object abstention; keep all GT in evaluation',
                       'finish_reason':response.get('finish_reason')})
            return {}


    def conflict_request(messages, _):
        from matching_compat import legacy_conflict
        data = request(messages, 'conflict', conflict_client)
        selected,raw = legacy_conflict(data)
        candidate_ids = {v for m in messages for c in m['content'] if c['type']=='text'
                         for v in re.findall(r"and id '([^']+)'", c['text'])}
        if str(selected).lower() == 'none':
            selected = 'none'
        elif selected not in candidate_ids:
            write_json(out/f'{request_count:03d}_conflict.abstained.json',
                       {'returned_id':selected,'allowed_ids':sorted(candidate_ids),
                        'policy':'Unknown ID cannot create a match; no extra model call'})
            selected = 'none'
        return selected,raw

    helpers['run_qwen_pred_vs_gt_inference'] = conflict_request
    candidates = [(str(n['id']), n['name'], n.get('description','')) for n in pred['nodes']]
    candidate_ids = {x[0] for x in candidates}
    def process_gt(node):
        gid, label = str(node['id']), node['label']
        image_dir = out/'images'/f'gt_{gid}'
        paths = helpers['prepare_images_with_selection'](scan, gid, label, cfg['data_root'], str(image_dir),
                                                       max_images=5, min_area=1000)
        if paths and candidates:
            messages = helpers['build_prompt_for_matching'](paths, label, gid, candidates)
            data = request(messages, f'gt_{gid}')
            from matching_compat import candidate_record
            ids, scores, note = candidate_record(data, candidate_ids)
            write_json(out/f'gt_{gid}.candidate_decode.json',note)
        else:
            ids, scores = ['none']*3, ([1]*3 if not paths else [0.0]*3)
        record = {'scan_id':scan, 'gt_obj_id':gid, 'gt_label':label,
                  'match_pred_node_ids':ids, 'similarity_scores':scores}
        write_json(out/f'gt_{gid}.candidates.json',record)
        print(f'Matched GT {gid} ({label}) -> {ids}, scores={scores}',flush=True)
        return record
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(process_gt, gt['nodes']))
    used = list({p for r in records for p in r['match_pred_node_ids'] if p!='none'})
    mapping = helpers['pred_vs_gt']({'nodes_mapping':records},used,candidates,None)
    # The legacy resolver initializes every GT ID, including explicitly unmatched ones.
    scored = {str(k):metrics(gt,pred,mapping,k) for k in [1,3]}
    write_json(out/'object_matching.json',{'scan_id':scan,'gt2pred':mapping,'nodes_mapping':records})
    pnodes = {str(n['id']):n for n in pred['nodes']}
    object_details = [{'gt_id':str(n['id']),'gt_label':n['label'],'pred_id':mapping[str(n['id'])],
                       'pred_node':pnodes.get(mapping[str(n['id'])])} for n in gt['nodes']]
    legacy = load_functions(legacy_eval)
    legacy_results = {}
    for k in [1,3]:
        capped = deepcopy(pred)
        for e in capped['edges']:e['c_edge']=e['c_edge'][:k]
        vals = legacy['evaluate_single_scan'](scan,gt,capped,mapping,True)
        assert vals[0]==len(gt['nodes']) and vals[1]==scored[str(k)]['object_hits']
        assert vals[4]==scored[str(k)]['gt_relation_rows']
        assert vals[5]==scored[str(k)]['both_endpoints_matched_rows']
        assert vals[6]==scored[str(k)]['relationship_hits']
        legacy_results[str(k)]={'object_total':vals[0],'object_hit':vals[1],
                               'edge_total_all':vals[4],'edge_total_both':vals[5],'edge_hit':vals[6]}
    kept = manifest.get('observed_rgb_frames', [p for f in run.glob('win*.audit.json') for p in json.loads(f.read_text())['api_images']])
    result = {'status':'completed','scope':'single_scan_diagnostic','scan':scan,
              'matching_model':client.config['model'],'temperature':0,'enable_thinking':False,
              'matching_api_calls':request_count,'external_api_calls':0,'local_model_requests':request_count,'predicate_judge':False,
              'object_details':object_details,'metrics':scored,
              'legacy_no_judge_cross_check':legacy_results,'visibility':visibility(gt,bboxes,kept),
              'limitations':['Object correspondence is local-model-judged and requires spot review.',
                             'One short scan cannot establish aggregate accuracy or baseline comparability.',
                             'Top K is per directed object pair, not a global confidence-ranked R@K.']}
    write_json(out/'results.json',result)
    print('Evaluation complete:',out/'results.json',flush=True)


if __name__=='__main__':main()
