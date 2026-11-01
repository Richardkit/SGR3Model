"""Read-only preflight by default; --execute runs local Qwen generation and matching."""
import argparse,json,os,shutil,sys
from pathlib import Path
from _shared import PACKAGE,load_config,write_new
sys.path[:0]=[str(PACKAGE),str(PACKAGE/'src')]
from evaluation.base import check_kb_members
from evaluation.evaluate import digest,evaluate
from graph_legacy import list_sequence_rgbs,plan_windows
from retrieval_cache import CachedRetrieval,read_sealed,signature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--model-path',default=os.environ.get('QWEN_MODEL_PATH'))
    p.add_argument('--scans',nargs='+');p.add_argument('--output-root',type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);ids=Path(cfg['evaluation_split']).read_text().split();selected=a.scans or ids
    if len(selected)!=len(set(selected)) or set(selected)-set(ids):raise ValueError('Invalid scan subset')
    kb=json.loads((Path(cfg['kb_dir'])/'manifest.json').read_text())
    if digest(cfg['evaluation_split'])!=kb['split_sha256'] or digest(cfg['scene_metadata'])!=kb['metadata_sha256']:
        raise ValueError('KB split/metadata provenance mismatch')
    isolation=check_kb_members(kb['included_scans'],ids,json.loads(Path(cfg['scene_metadata']).read_text()))
    if not isolation['passed']:raise ValueError('KB contains test scene family members')
    cache=Path(cfg['retrieval_cache_dir'])/'manifest.json'
    plan={'execute':a.execute,'scans':selected,'scene_isolation':isolation,'model_loaded':False,
          'image_policy':'original full window if retained queries nonempty','patch_topk':10,
          'retrieval_selection':'global cumulative frame score Top5, no scene argmax'}
    if not cache.exists():
        plan['status']='needs_new_k10_retrieval_cache';print(json.dumps(plan,ensure_ascii=False),flush=True)
        if a.execute:raise ValueError('Build the new paper k=10 cache first')
        return
    manifest=read_sealed(cache)
    if manifest['signature']!=signature(cfg):raise ValueError('Wrong configuration in retrieval cache')
    for sid in selected:
        frames=list_sequence_rgbs(str(Path(cfg['data_root'])/sid/'sequence'))
        c=CachedRetrieval(cfg,sid,frames)
        for wi,(s,t) in enumerate(plan_windows(len(frames),6,4)):
            row=c.row(wi,frames[s:t])
            if wi<2 and row['kept_images']!=frames[s:t]:raise ValueError('Warmup windows must bypass filtering')
            if row['retrieval'] is not None and row['retrieval'].get('selection_policy')!='global_frame_sum_topk_v1':
                raise ValueError('Unexpected retrieval algorithm')
    if not a.model_path:raise ValueError('Provide --model-path or QWEN_MODEL_PATH')
    model=Path(a.model_path).resolve();mc=json.loads((model/'config.json').read_text())
    if mc.get('model_type')!='qwen3_vl' or mc.get('text_config',{}).get('num_hidden_layers')!=64:
        raise ValueError('Expected the local Qwen3-VL-32B checkpoint')
    wi=model/'model.safetensors.index.json';shards=set(json.loads(wi.read_text())['weight_map'].values())
    if not all((model/s).is_file() for s in shards):raise ValueError('Missing model weight shards')
    plan.update(status='preflight_passed',model_path=str(model),weight_shards=len(shards),
                source_sha256={str(p):digest(p) for p in [Path(a.config),cache,model/'config.json',wi,*list((PACKAGE/'src').glob('*.py')),*list((PACKAGE/'legacy_matching').glob('*.py'))]})
    print(json.dumps(plan,ensure_ascii=False),flush=True)
    if not a.execute:return
    if a.output_root is None:raise ValueError('--output-root is required')
    out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=False)
    for folder in ['src','legacy_matching']:shutil.copytree(PACKAGE/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__'))
    for folder in ['configs','reports','runs']:(out/folder).mkdir()
    write_new(out/'configs/config.json',cfg);write_new(out/'reports/preflight.json',plan)
    (out/'reports/model_path.txt').write_text(str(model)+'\n')
    # Materialized imports bind all output and matching-source paths to this new run.
    for name in ['common','graph_legacy','retrieval_cache','run','persistent_client','local_vllm_client',
                 'local_matching_client','evaluate_single_scan_local','batch_client','soft_output','matching_compat']:
        sys.modules.pop(name,None)
    sys.path.insert(0,str(out/'src'))
    os.environ.pop('SGR3_API_RESPONSE_REUSE_FILE',None)
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    from persistent_client import GenerationClient,MatchingClient
    import persistent_client as pool
    import run as generation
    import evaluate_single_scan_local as matching
    generation.DashScopeClient=GenerationClient;matching.DashScopeClient=MatchingClient
    engine=GenerationClient(cfg['api']).llm
    states={s:{'state':'pending'} for s in ids}
    for sid in selected:
        frames=list_sequence_rgbs(str(Path(cfg['data_root'])/sid/'sequence'));n=len(plan_windows(len(frames),6,4))
        rid='scan-'+sid;folder=out/'runs'/rid
        generation.main(['--config',str(out/'configs/config.json'),'--method','sgr3','--scan',sid,
                         '--run-id',rid,'--max-windows',str(n),'--max-calls',str(2*n+1)])
        matching.main(['--run',str(folder),'--output-name','object_matching'])
        if pool._ENGINE is not engine:raise RuntimeError('Model unexpectedly reloaded')
        states[sid]={'state':'complete','prediction':str(folder/'graph_final.json'),
                     'mapping':str(folder/'object_matching/object_matching.json')}
        m={'method':'SGR3_paper_globalframes_k10_scene_excluded','split':cfg['evaluation_split'],
           'GT_root':cfg['data_root'],'scans':states.copy()}
        mp=out/'reports'/f'manifest_{len([x for x in states.values() if x["state"]=="complete"]):02d}.json'
        write_new(mp,m);write_new(mp.with_name(mp.stem+'_scores.json'),evaluate(mp))
    write_new(out/'reports/COMPLETED_REQUESTED_SCANS.json',{'selected_scans':selected,'full47':len(selected)==len(ids),'engine_initialized_once':True})


if __name__=='__main__':main()
