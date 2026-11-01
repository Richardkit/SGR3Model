"""Compute all fixed test-window retrievals before local Qwen execution; checkpoint every window."""
import argparse,gc,json,os,pickle,sys,time
from pathlib import Path
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'src'))
from common import sha256,write_json,retrieval_exclusions,scene_map
from graph_legacy import list_sequence_rgbs,plan_windows
from retrieval_cache import seal,read_sealed,signature
from _shared import load_config


def compute(config_path):
    from prepared_scan import PreparedScan
    import faiss
    import vision_legacy as vision
    cfg=load_config(config_path);arm='paper_globalframes_k10';root=Path(cfg['retrieval_cache_dir']);root.mkdir(parents=True,exist_ok=True)
    sig=signature(cfg);mp=root/'manifest.json'
    sources={str(p):sha256(p) for p in [Path(__file__),E/'src/vision_legacy.py',E/'src/prepared_scan.py',E/'src/retrieval_cache.py',E/'src/common.py',E/'src/graph_legacy.py']}
    prior=read_sealed(mp) if mp.exists() else None
    if prior and (prior['signature']!=sig or prior['producer_source_sha256']!=sources):raise RuntimeError('Frozen cache source/config changed')
    if prior and prior['status']=='complete':
        print(arm,'cache already complete',flush=True);return
    kb=Path(cfg['kb_dir']);km=json.loads((kb/'manifest.json').read_text());ids=Path(cfg['evaluation_split']).read_text().split();metadata=json.loads(Path(cfg['scene_metadata']).read_text());blocked=retrieval_exclusions(metadata,ids,cfg['retrieval_exclusion_policy']);known=set(scene_map(metadata))
    if km.get('retrieval_exclusion_policy','scene')!=cfg['retrieval_exclusion_policy'] or km['split_sha256']!=sha256(cfg['evaluation_split']) or km['metadata_sha256']!=sha256(cfg['scene_metadata']) or set(km['included_scans']) & blocked:raise RuntimeError('KB provenance/policy mismatch')
    for name,digest in km['artifacts'].items():
        if sha256(kb/name)!=digest:raise RuntimeError('KB hash mismatch: '+name)
    faiss.omp_set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    faiss.cvar.distance_compute_blas_threshold=cfg['faiss_blas_threshold']
    index=faiss.read_index(str(kb/'index.faiss'),faiss.IO_FLAG_MMAP|faiss.IO_FLAG_READ_ONLY);index.nprobe=cfg['nprobe']
    with (kb/'keys.pkl').open('rb') as f:keys=pickle.load(f)
    keys=[k.decode('utf-8','ignore') if isinstance(k,(bytes,bytearray)) else str(k) for k in keys]
    if len(keys)!=index.ntotal:raise RuntimeError('Index/key mismatch')
    key_scans={vision.parse_kb_key_to_scan_frame(k)[0] for k in keys}
    if key_scans & blocked or key_scans-known:raise RuntimeError('Forbidden/unknown scan in index keys')
    vision.RETRIEVAL_TEMPERATURE=cfg['retrieval_temperature'];vision.UNWEIGHTED=False
    state=dict(status='building',signature=sig,producer_source_sha256=sources,kb_manifest=km,
               kb_manifest_sha256=sha256(kb/'manifest.json'),index_key_scan_count=len(key_scans),excluded_index_scans=[],
               input_sha256={k:sha256(cfg[k]) for k in ['evaluation_split','scene_metadata','classes','predicates']},scans={},retained_windows=0,planned_windows=0)
    write_json(mp,seal(state));started=time.time();parity_done=False
    for sid in ids:
        folder=root/sid;folder.mkdir(exist_ok=True);frames=list_sequence_rgbs(str(Path(cfg['data_root'])/sid/'sequence'));prepared=PreparedScan(cfg,sid,frames)
        rows=[];plan=plan_windows(len(frames),cfg['window_size'],cfg['stride'])
        for wi,(start,end) in enumerate(plan):
            original=frames[start:end];kept=prepared.kept_images(wi,original);path=folder/f'win{wi:04d}.json'
            if path.exists():
                row=read_sealed(path)
                if row['signature']!=sig or row['scan']!=sid or row['window']!=wi or row['original_images']!=original or row['kept_images']!=kept:raise RuntimeError('Existing cache identity mismatch')
                for name,digest in row['reference_sources_sha256'].items():
                    if sha256(name)!=digest:raise RuntimeError('Reference graph changed')
            else:
                r=None;reference_hashes={}
                if kept:
                    raw=folder/f'win{wi:04d}.retrieval.json'
                    vision.run_rag_for_window(str(Path(cfg['data_root'])/sid/'sequence'),index,keys,cfg['data_root'],kept,None,str(raw),faiss_topk=cfg['reference_frame_topk'],patch_topk=cfg['patch_topk'],excluded_scan_ids=blocked,known_scan_ids=known,query_patch_cache=prepared.query_patch_cache,keys_are_normalized=True)
                    r=json.loads(raw.read_text());reference_hashes={f['json_path']:sha256(f['json_path']) for f in r['loaded_reference_frames']}
                    if not parity_done:
                        check=folder/'key_normalization_parity.json'
                        legacy=vision.run_rag_for_window(str(Path(cfg['data_root'])/sid/'sequence'),index,keys,cfg['data_root'],kept,None,str(check),faiss_topk=cfg['reference_frame_topk'],patch_topk=cfg['patch_topk'],excluded_scan_ids=blocked,known_scan_ids=known,query_patch_cache=prepared.query_patch_cache)
                        if legacy!=r['reference_edges'] or json.loads(check.read_text())!=r:raise RuntimeError('Key normalization optimization changed retrieval')
                        write_json(root/'parity.json',dict(passed=True,scan=sid,window=wi,exact_reference_edges=True,exact_frame_scores=True));parity_done=True
                row=seal(dict(signature=sig,scan=sid,window=wi,original_images=original,kept_images=kept,retrieval=r,reference_sources_sha256=reference_hashes))
                write_json(path,row)
            rows.append(dict(file=path.name,sha256=sha256(path)));state['planned_windows']+=1;state['retained_windows']+=bool(kept)
            write_json(root/'progress.json',dict(arm=arm,status='building',scans_complete=len(state['scans']),scan=sid,window=wi,planned_windows=state['planned_windows'],retained_windows=state['retained_windows'],elapsed_seconds=time.time()-started))
        scan=seal(dict(scan=sid,frames=frames,image_sha256={p:sha256(p) for p in frames},precomputed_provenance=prepared.provenance,windows=rows))
        write_json(folder/'manifest.json',scan);state['scans'][sid]=sha256(folder/'manifest.json');write_json(mp,seal(state));print(arm,len(state['scans']),'of',len(ids),'scans complete',flush=True)
        del prepared;gc.collect()
    state['status']='complete';state['finished_at']=time.time();write_json(mp,seal(state));write_json(root/'progress.json',dict(arm=arm,status='complete',scans_complete=len(ids),planned_windows=state['planned_windows'],retained_windows=state['retained_windows'],elapsed_seconds=time.time()-started))
    print(arm,'COMPLETE',state['retained_windows'],'retained windows',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    if a.execute:compute(a.config)
    else:print(json.dumps({'execute':False,'index_training':False,'cache':cfg['retrieval_cache_dir'],'patch_topk':cfg['patch_topk'],'selection':cfg['retrieval_selection_policy']}))
