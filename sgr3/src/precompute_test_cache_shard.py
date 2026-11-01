"""GPU-only preprocessing: frozen ColQwen decisions and exact SigLIP query patches; no API."""
import argparse
from collections import Counter
import json,os,time,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import sha256,write_json
from graph_legacy import list_sequence_rgbs,plan_windows
import filter_legacy as filtering
import vision_legacy as vision

ROOT=Path(__file__).resolve().parents[1]


def make_signature(cfg):
    return {k:cfg[k] for k in ['window_size','stride','filter_warmup_windows','colqwen_threshold',
                              'colqwen_model','colqwen_revision','siglip_model','siglip_revision']}


def prefilter_candidates(tokens,paths):
    proxy=filtering.cheap_image_proxy(tokens,mode='mean')
    idx,sims=filtering.topM_by_cosine(proxy,M=30)
    return idx,sims


def candidate_indices(index,candidate_idx,candidate_sim,buffer_indices):
    pairs=[(int(j),float(s)) for j,s in zip(candidate_idx[index],candidate_sim[index])
           if j in buffer_indices and s>=0.45]
    pairs.sort(key=lambda x:x[1],reverse=True)
    return [j for j,_ in pairs[:5]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--scans-file',required=True);p.add_argument('--limit-scans',type=int);p.add_argument('--guard-band',type=float,default=.03)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.output).resolve()
    out.mkdir(exist_ok=False,parents=True)
    ids=Path(cfg['evaluation_split']).read_text().split()
    inventory=[]
    for sid in ids:
        frames=list_sequence_rgbs(str(Path(cfg['data_root'])/sid/'sequence'))
        if not frames:raise ValueError('No images: '+sid)
        inventory.append((sid,frames))
    selected=Path(args.scans_file).read_text().split()
    if len(selected)!=len(set(selected)) or not set(selected)<=set(ids):raise ValueError('Invalid shard scan list')
    inventory=[row for row in inventory if row[0] in set(selected)]
    inventory.sort(key=lambda row:(len(row[1]),row[0]))
    if args.limit_scans:inventory=inventory[:args.limit_scans]
    manifest={'status':'running','config_signature':make_signature(cfg),'split_sha256':sha256(cfg['evaluation_split']),
      'scans':[s for s,_ in inventory],'planned_scans':len(inventory),'planned_frames':sum(len(f) for _,f in inventory),
      'gpu_job_id':os.environ.get('SLURM_JOB_ID'),'source_sha256':{p.name:sha256(p) for p in [Path(__file__),ROOT/'src/filter_legacy.py',ROOT/'src/vision_legacy.py']},
      'optimization':'Each ColQwen image encoded separately and reused; same mean-MaxSim formula; near-threshold results recomputed using original candidate batch.',
      'guard_band':args.guard_band,'pre_filter':{'proxy':'mean','top_m':30,'min_similarity':.45,'topk_buffer':5},
      'threshold_rule':'keep iff max_similarity < threshold; first two windows bypass; buffer updated after complete window',
      'completed_scans':[]}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in [Path(__file__),ROOT/'src/filter_legacy.py',ROOT/'src/vision_legacy.py']:
        shutil.copy2(path,snapshot/path.name)
    write_json(out/'manifest.json',manifest)
    filtering.COLQWEN_REVISION=cfg['colqwen_revision'];filtering.DATA_ROOT=cfg['data_root']
    processor,model=filtering.load_colqwen(cfg['colqwen_model']);manifest['adapter_validation']=model._robio_adapter_validation
    manifest['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
    manifest['gpu_uuid']=str(getattr(torch.cuda.get_device_properties(0),'uuid','unknown'))
    manifest['gpu_name']=torch.cuda.get_device_name(0)
    vision.SIGLIP_REVISION=cfg['siglip_revision'];siglip=vision.load_siglip(cfg['siglip_model'])
    started=time.monotonic()
    for number,(sid,frames) in enumerate(inventory):
        scan_start=time.monotonic();folder=out/sid;folder.mkdir(exist_ok=False)
        sigdir=Path(cfg['data_root'])/sid/'siglip2_val';token_path=sigdir/'patch_tokens_fp16.npy';path_file=sigdir/'paths.json'
        tokens=np.load(token_path,mmap_mode='r');proxy_paths=[Path(x).name for x in json.loads(path_file.read_text())]
        assert len(proxy_paths)==len(tokens) and len(set(proxy_paths))==len(proxy_paths)
        positions={name:i for i,name in enumerate(proxy_paths)};byname={Path(f).name:f for f in frames}
        assert set(proxy_paths)<=set(byname)
        neighbors,similarities=prefilter_candidates(tokens,proxy_paths)
        encodings={};scores_cache={};buffer=[];windows=[];guarded=0;sampled=0;max_drift=0.;disagreements=0
        def encode(path):
            if path not in encodings:
                with torch.inference_mode():
                    inp=processor.process_images([Image.open(path).convert('RGB')]).to(model.device)
                    value=model(**inp).detach().cpu()[0].contiguous()
                encodings[path]=value
            return encodings[path]
        def score(query,candidates):
            nonlocal guarded,sampled,max_drift,disagreements
            key=(query,tuple(candidates))
            if key in scores_cache:return scores_cache[key]
            q=encode(query);cs=[encode(f) for f in candidates]
            with torch.inference_mode():
                raw=processor.score_multi_vector([q],cs)[0]
            cached=(raw/q.shape[0]).tolist();maximum=max(cached)
            # Validate two nonborderline comparisons per scan as well as every near-threshold case.
            near=abs(maximum-cfg['colqwen_threshold'])<=args.guard_band
            check=near or sampled<2
            if check:
                exact=filtering.colqwen_score_image_to_images(processor,model,query,candidates)
                drift=max(abs(a-b) for a,b in zip(cached,exact));max_drift=max(max_drift,drift)
                disagrees=(max(cached)<cfg['colqwen_threshold'])!=(max(exact)<cfg['colqwen_threshold'])
                disagreements+=int(disagrees)
                if not near and disagrees:raise RuntimeError('Cached score crosses threshold outside the guard band')
                if drift>args.guard_band:raise RuntimeError('ColQwen numeric drift exceeds guard band; review before full caching')
                cached=exact
                guarded+=int(near);sampled+=int(not near)
            scores_cache[key]=cached
            return cached
        for wi,(start,end) in enumerate(plan_windows(len(frames),cfg['window_size'],cfg['stride'])):
            original=frames[start:end];kept=[];decisions=[]
            if wi<cfg['filter_warmup_windows']:
                kept=list(original)
            else:
                buffer_indices={positions[Path(f).name] for f in buffer if Path(f).name in positions}
                for frame in original:
                    pi=positions.get(Path(frame).name)
                    candidates=[] if pi is None else [byname[proxy_paths[j]] for j in candidate_indices(pi,neighbors,similarities,buffer_indices)]
                    values=score(frame,candidates) if candidates else []
                    maximum=max(values) if values else None
                    keep=maximum is None or maximum<cfg['colqwen_threshold']
                    if keep:kept.append(frame)
                    decisions.append({'frame':Path(frame).name,'candidates':[Path(f).name for f in candidates],
                                      'scores':values,'max_similarity':maximum,'keep':keep})
            for frame in kept:
                if frame not in buffer:buffer.append(frame)
            windows.append({'window':wi,'start':start,'end':end,'original_images':[Path(f).name for f in original],
                            'kept_images':[Path(f).name for f in kept],'decisions':decisions})
        # Check every kept list against the already executed original-window pilot when available.
        pilot=ROOT/'runs/original-scene-112949'
        if sid=='20c993b7-698f-29c5-847d-c8cb8a685f5a' and (pilot/'status.json').exists():
            for w in windows:
                old=json.loads((pilot/f"win{w['window']:04d}.audit.json").read_text())
                assert w['kept_images']==[Path(f).name for f in old['kept_images']], 'Pilot filter masks differ'
        write_json(folder/'filter_windows.json',windows)
        torch.save({Path(k).name:v for k,v in encodings.items()},folder/'colqwen_embeddings.pt')
        # Query features must use the actual 24-patch retrieval encoder, not the legacy 195-token proxy cache.
        patches=[]
        with torch.inference_mode():
            for frame in frames:
                patches.append(vision.embed_image_patches_siglip2(siglip,Image.open(frame).convert('RGB')))
        values=np.stack(patches);assert values.shape==(len(frames),24,768) and np.isfinite(values).all()
        np.save(folder/'query_patches.npy',values,allow_pickle=False)
        write_json(folder/'frames.json',[Path(f).name for f in frames])
        info={'status':'complete','scan':sid,'frames':len(frames),'windows':len(windows),
          'config_signature':make_signature(cfg),'image_sha256':{Path(f).name:sha256(f) for f in frames},
          'proxy_inputs_sha256':{str(p):sha256(p) for p in [token_path,path_file]},
          'kept_windows':sum(bool(w['kept_images']) for w in windows),'kept_image_presentations':sum(len(w['kept_images']) for w in windows),
          'encoded_colqwen_images':len(encodings),'guarded_comparisons':guarded,'sampled_comparisons':sampled,
          'max_verified_score_drift':max_drift,'repaired_threshold_disagreements':disagreements,
          'seconds':time.monotonic()-scan_start,
          'artifacts':{n:sha256(folder/n) for n in ['filter_windows.json','colqwen_embeddings.pt','query_patches.npy','frames.json']}}
        write_json(folder/'manifest.json',info)
        manifest['completed_scans'].append(sid);manifest['elapsed_seconds']=time.monotonic()-started
        write_json(out/'manifest.json',manifest)
        print(f"Completed {number+1}/{len(inventory)} {sid}: {len(frames)} frames, {info['seconds']:.1f}s, kept windows {info['kept_windows']}/{len(windows)}, guarded {guarded}",flush=True)
    manifest['status']='complete';manifest['elapsed_seconds']=time.monotonic()-started
    manifest['scan_manifests_sha256']={sid:sha256(out/sid/'manifest.json') for sid,_ in inventory}
    write_json(out/'manifest.json',manifest);print('Complete:',out,flush=True)

if __name__=='__main__':main()
