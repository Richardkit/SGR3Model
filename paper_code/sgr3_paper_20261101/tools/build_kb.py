"""Optional new scene-disjoint index build from existing SigLIP embeddings; default inspect only."""
import argparse,json,subprocess,sys
from pathlib import Path
from _shared import PACKAGE,load_config,write_new

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--nlist',type=int,default=4096)
    p.add_argument('--training-size',type=int,default=300000);p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);cfg['kb_dir']=str(a.output.resolve())
    print(json.dumps({'execute':a.execute,'new_index':cfg['kb_dir'],'requires_GPU':False,'scene_exclusion':True}),flush=True)
    if a.execute:
        if a.output.exists():raise FileExistsError('Use a new output directory')
        resolved=a.output.with_name(a.output.name+'.config.json');write_new(resolved,cfg)
        subprocess.run([sys.executable,str(PACKAGE/'src/build_kb.py'),'--config',str(resolved),
                        '--nlist',str(a.nlist),'--training-size',str(a.training_size)],check=True)
