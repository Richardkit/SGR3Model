import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import time
from pathlib import Path

from common import (add_votes, decode_predicate_ids, normalize_delta, excluded_scans, filtered_messages, ranked_graph,
                    scene_map, sha256, validate_delta, write_json, retrieval_exclusions, select_generation_images)
from dashscope_client import parse_json
from local_vllm_client import LocalVLLMClient as DashScopeClient
from soft_output import parse_model_output
from graph_legacy import list_sequence_rgbs, load_vocab_list, merge_graphs, plan_windows

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'configs/ready_pilot.json'))
    p.add_argument('--method', choices=['sgr3'], default='sgr3')
    p.add_argument('--scan', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--max-windows', type=int, default=3)
    p.add_argument('--max-calls', type=int, default=3)
    args = p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) or args.max_windows < 1 or args.max_calls < 1:
        p.error('Use a simple run ID and positive explicit limits')
    cfg = json.loads(Path(args.config).read_text())
    if cfg.get('retrieval_cache_dir') and args.method != 'sgr3':
        p.error('This retrieval cache is frozen for the SGR3 paired experiment')
    eval_ids = Path(cfg['evaluation_split']).read_text().split()
    if args.scan not in eval_ids:
        p.error('Scan is not in the frozen evaluation split')
    client = DashScopeClient(cfg['api'])
    cfg['api'] = client.config
    use_filter = args.method not in {'only_qwen', 'no_filter'}
    use_rag = args.method not in {'only_qwen', 'no_rag'}
    frames = list_sequence_rgbs(str(Path(cfg['data_root']) / args.scan / 'sequence'))
    if not frames:
        raise RuntimeError('No input RGBs')
    blocked = retrieval_exclusions(json.loads(Path(cfg['scene_metadata']).read_text()), eval_ids,
                                   cfg.get('retrieval_exclusion_policy', 'scene'))
    kb_manifest = None
    if use_rag and cfg.get('retrieval_cache_dir'):
        from retrieval_cache import CachedRetrieval
        kb_manifest = CachedRetrieval(cfg, args.scan, frames).manifest['kb_manifest']
    elif use_rag:
        kb = Path(cfg['kb_dir'])
        kb_manifest = json.loads((kb / 'manifest.json').read_text())
        if kb_manifest.get('retrieval_exclusion_policy', 'scene') != cfg.get('retrieval_exclusion_policy', 'scene'):
            raise RuntimeError('KB exclusion policy mismatch')
        if kb_manifest['split_sha256'] != sha256(cfg['evaluation_split']) or kb_manifest['metadata_sha256'] != sha256(cfg['scene_metadata']):
            raise RuntimeError('KB split/metadata mismatch')
        if set(kb_manifest['included_scans']) & blocked:
            raise RuntimeError('KB contains an evaluation scene')
        for name, digest in kb_manifest['artifacts'].items():
            if sha256(kb / name) != digest:
                raise RuntimeError('KB artifact hash mismatch: ' + name)
    out = ROOT / 'runs' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    source_snapshot = out / 'source_snapshot'
    source_snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, source_snapshot / source.name)
    packages = {}
    for name in ['torch', 'transformers', 'colpali-engine', 'faiss-cpu', 'numpy', 'Pillow']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(out / 'manifest.json', {'config': cfg, 'arguments': vars(args), 'python': platform.python_version(),
        'packages': packages, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'code_sha256': {s.name: sha256(s) for s in Path(__file__).parent.glob('*.py')},
        'input_sha256': {key: sha256(cfg[key]) for key in ['evaluation_split', 'scene_metadata', 'classes', 'predicates']},
        'kb_manifest': kb_manifest,
        'retrieval_cache_manifest_sha256': sha256(Path(cfg['retrieval_cache_dir'])/'manifest.json') if cfg.get('retrieval_cache_dir') else None,
        'api_response_reuse_file_sha256': sha256(os.environ['SGR3_API_RESPONSE_REUSE_FILE']) if os.environ.get('SGR3_API_RESPONSE_REUSE_FILE') else None,
        'generation_backend':'local vLLM0.11.2 Qwen3-VL-32B-Instruct; api config retained for cache signature and fixed matching only',
        'api_call_accounting': 'Logical calls include exact-request reuse; response_reuse identifies reused replies without new HTTP requests'})
    try:
        execute(args, cfg, client, frames, out, use_filter, use_rag, blocked)
    except Exception as e:
        write_json(out / 'status.json', {'status': 'failed', 'error_type': type(e).__name__,
                                        'message': str(e) if isinstance(e, (ValueError, RuntimeError)) else 'See job traceback'})
        raise


def execute(args, cfg, client, frames, out, use_filter, use_rag, blocked):
    prepared = None
    if use_rag and cfg.get('retrieval_cache_dir'):
        from retrieval_cache import CachedRetrieval
        prepared = CachedRetrieval(cfg, args.scan, frames)
        write_json(out / 'precomputed_inputs.json', prepared.provenance)
    elif cfg.get('precomputed_cache_dir'):
        from prepared_scan import PreparedScan
        prepared = PreparedScan(cfg, args.scan, frames)
        write_json(out / 'precomputed_inputs.json', prepared.provenance)
    if (use_filter or use_rag) and prepared is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('ColQwen/retrieval pilot requires a GPU allocation')
    if use_filter and prepared is None:
        import filter_legacy as filtering
        filtering.DATA_ROOT = cfg['data_root']
        filtering.COLQWEN_REVISION = cfg.get('colqwen_revision')
        processor, model = filtering.load_colqwen(cfg['colqwen_model'])
    if use_rag and not cfg.get('retrieval_cache_dir'):
        import faiss
        import pickle
        import vision_legacy as vision
        kb = Path(cfg['kb_dir'])
        if cfg.get('index_io_mode') == 'read_only_memory_mapping':
            index = faiss.read_index(str(kb / 'index.faiss'), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        else:
            index = faiss.read_index(str(kb / 'index.faiss'))
        index.nprobe = cfg['nprobe']
        with (kb / 'keys.pkl').open('rb') as f:
            keys = pickle.load(f)
        if index.ntotal != len(keys):
            raise RuntimeError('Index/key count mismatch')
        # Independently check every identity used by retrieval; never trust the directory name.
        if any(vision.parse_kb_key_to_scan_frame(k)[0] in blocked for k in keys):
            raise RuntimeError('Evaluation scene found in KB keys')
        known_scans = set(scene_map(json.loads(Path(cfg['scene_metadata']).read_text())))
        if any(vision.parse_kb_key_to_scan_frame(k)[0] not in known_scans for k in keys):
            raise RuntimeError('Unknown scan found in KB keys')
        vision.RETRIEVAL_TEMPERATURE = cfg['retrieval_temperature']
        vision.SIGLIP_REVISION = cfg.get('siglip_revision')
        vision.UNWEIGHTED = args.method == 'unweighted'
        siglip = None if prepared is not None else vision.load_siglip(cfg['siglip_model'])
    classes, predicates = load_vocab_list(cfg['classes']), load_vocab_list(cfg['predicates'])
    graph = {'nodes': [], 'edges': []}
    votes, buffer = {}, []
    plan = plan_windows(len(frames), cfg['window_size'], cfg['stride'])
    calls, processed = 0, 0
    for wi, (start, end) in enumerate(plan[:args.max_windows]):
        if calls >= args.max_calls:
            break
        original = frames[start:end]
        kept = list(original)
        begin = time.monotonic()
        if use_filter and prepared is not None:
            kept = prepared.kept_images(wi, original)
        elif use_filter and wi >= cfg['filter_warmup_windows']:
            kept = filtering.img_if_new_check_colqwen(original, args.scan, buffer, model, processor,
                                                     COLQWEN_SIM_THRESH=cfg['colqwen_threshold'])
        if not isinstance(kept, list) or len(set(kept)) != len(kept) or not set(kept) <= set(original):
            raise ValueError('Invalid filter output; refusing retrieval/generation')
        audit = {'window': wi, 'original_images': original, 'kept_images': kept,
                 'filter_enabled': use_filter, 'filter_applied': use_filter and wi >= cfg['filter_warmup_windows'],
                 'filter_seconds': time.monotonic() - begin, 'api_images': []}
        prefix = out / f'win{wi:04d}'
        processed += 1
        if not kept:
            audit['status'] = 'skipped_no_new_images'
            write_json(str(prefix) + '.audit.json', audit)
            print(f'Window {wi + 1}/{len(plan)}: input={len(original)} kept=0; skipped', flush=True)
            continue
        for image in kept:
            if image not in buffer:
                buffer.append(image)
        references = []
        if use_rag and cfg.get('retrieval_cache_dir'):
            references = prepared.retrieve(wi, original, kept, str(prefix)+'.retrieval.json', blocked)
        elif use_rag:
            references = vision.run_rag_for_window(
                str(Path(cfg['data_root']) / args.scan / 'sequence'), index, keys, cfg['data_root'],
                kept, siglip, str(prefix) + '.retrieval.json',
                faiss_topk=cfg.get('reference_frame_topk', 5), patch_topk=cfg.get('patch_topk', 10),
                excluded_scan_ids=blocked, known_scan_ids=known_scans,
                query_patch_cache=prepared.query_patch_cache if prepared is not None else None)
        messages = filtered_messages(original, kept, graph, references, classes, predicates,
                                     predicate_encoding=cfg.get('predicate_encoding', 'labels'),
                                     image_policy=cfg.get('generation_image_policy', 'filtered'),
                                     coverage_policy=cfg.get('generation_coverage_policy', 'legacy'))
        api_images = select_generation_images(original, kept, cfg.get('generation_image_policy', 'filtered'))
        audit['generation_image_policy'] = cfg.get('generation_image_policy', 'filtered')
        audit['generation_coverage_policy'] = cfg.get('generation_coverage_policy', 'legacy')
        audit['api_images'] = api_images
        audit['image_sha256'] = {s: sha256(s) for s in api_images}
        audit['status'] = 'request_prepared'
        write_json(str(prefix) + '.audit.json', audit)
        # Store local image paths and prompt text; never store the Key or base64 image payload.
        write_json(str(prefix) + '.messages.json', messages)
        calls += 1
        try:
            result = client.complete(messages)
            write_json(str(prefix) + '.response.json', result)
            delta, recovery_notes = parse_model_output(result)
            audit['soft_response_recovery'] = recovery_notes
            write_json(str(prefix) + '.raw_delta.json', delta)
            if cfg.get('predicate_encoding') == 'ids':
                delta = normalize_delta(delta, graph, predicates)
            local = validate_delta(delta, graph, classes, predicates)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            if isinstance(error,RuntimeError) and any(x in str(error) for x in ['HTTP 401','HTTP 403','HTTP 404','EngineCore','CUDA error','out of memory','EngineDead']):raise
            audit.update(status='skipped_unusable_generation',soft_warning=str(error))
            write_json(str(prefix) + '.soft_error.json', {'type':type(error).__name__,'message':str(error),'policy':'Keep prior graph; no fabricated prediction; all GT retained in evaluation'})
            write_json(str(prefix) + '.audit.json', audit)
            print(f'Window {wi+1}/{len(plan)}: unusable output recorded; prior graph retained',flush=True)
            continue
        if cfg.get('coverage_completion', False):
            if calls >= args.max_calls:raise RuntimeError('No remaining coverage-call budget')
            from coverage_completion import complete_pairs
            try:
                delta, extra_calls = complete_pairs(client, messages, graph, delta, classes, predicates,
                                                   cfg.get('predicate_encoding','labels'),prefix)
            except (ValueError,KeyError,TypeError,RuntimeError) as error:
                extra_calls=int(Path(str(prefix)+'.completion.messages.json').exists())
                audit['coverage_completion_failed']=True
                write_json(str(prefix)+'.completion.failed.json',{'type':type(error).__name__,'message':str(error),'policy':'Preserve primary prediction'})
            calls += extra_calls
            audit['coverage_completion_calls']=extra_calls
            local=validate_delta(delta,graph,classes,predicates)
        audit['oov_node_labels']=[n['name'] for n in delta['new_nodes'] if n['name'] not in classes]
        audit['soft_events']=delta.get('_soft_events',[])
        audit['raw_proposed_nodes'] = len(delta['new_nodes'])
        audit['raw_proposed_pairs'] = len(delta['new_edges'])
        write_json(str(prefix) + '.delta.json', delta)
        add_votes(votes, delta['new_edges'])
        graph = merge_graphs(graph, local)
        write_json(str(prefix) + '.graph_union.json', graph)
        write_json(str(prefix) + '.graph_top3.json', ranked_graph(graph, votes, 3))
        write_json(str(prefix) + '.graph_ranked.json', ranked_graph(graph, votes, cfg['final_candidates']))
        audit['status'] = 'completed'
        write_json(str(prefix) + '.audit.json', audit)
        print(f'Window {wi + 1}/{len(plan)}: input={len(original)} kept={len(kept)} '
              f'nodes={len(graph["nodes"])} edges={len(graph["edges"])}', flush=True)
    complete = processed == len(plan)
    name = 'graph_final' if complete else 'graph_partial'
    write_json(out / (name + '.json'), ranked_graph(graph, votes, cfg['final_candidates']))
    write_json(out / (name + '_union.json'), graph)
    write_json(out / 'status.json', {'status': 'complete' if complete else 'pilot_partial',
        'planned_windows': len(plan), 'processed_windows': processed, 'api_calls': calls,
        'warning_windows':[json.loads(p.read_text())['window'] for p in out.glob('win*.audit.json') if json.loads(p.read_text()).get('soft_warning') or json.loads(p.read_text()).get('coverage_completion_failed')]} )


if __name__ == '__main__':
    main()
