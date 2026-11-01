"""Optional original SigLIP embedding producer with official scene-family exclusion."""
import argparse,importlib.util,json,sys
from pathlib import Path
from _shared import PACKAGE,load_config,write_new
sys.path[:0]=[str(PACKAGE),str(PACKAGE/'src')]
from common import excluded_scans,scene_map

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);meta=json.loads(Path(cfg['scene_metadata']).read_text())
    blocked=excluded_scans(meta,Path(cfg['evaluation_split']).read_text().split());known=set(scene_map(meta))
    unknown=[s.name for s in Path(cfg['data_root']).iterdir() if s.is_dir() and (s/'frame_sg').is_dir() and s.name not in known]
    if unknown:raise ValueError('Reference scan missing from official scene metadata: '+str(unknown))
    print(json.dumps({'execute':a.execute,'excluded_scene_scans':len(blocked),'output':str(a.output),'grid':[4,6]}),flush=True)
    if a.execute:
        a.output.mkdir(parents=True,exist_ok=False)
        blocked_path=a.output/'excluded_scans.txt';blocked_path.write_text('\n'.join(sorted(blocked))+'\n')
        spec=importlib.util.spec_from_file_location('reference_encoder',PACKAGE/'src/embedding_reference.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ROOT=Path(cfg['data_root']);module.OUTPUT_DIR=a.output;module.VAL_DIR=blocked_path
        module.MODEL_CKPT=cfg['siglip_model'];module.MODEL_REVISION=cfg['siglip_revision']
        sys.argv=['embedding_reference','--shard_id','0','--num_shards','1']
        module.main()
        write_new(a.output/'embedding_provenance.json',{'config':cfg,'excluded_scene_scans':sorted(blocked),
                  'producer':str(PACKAGE/'src/embedding_reference.py'),'original_encoder_grid':[4,6]})
