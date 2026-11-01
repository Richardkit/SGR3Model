"""Optional GPU feature/filter preparation. Existing checked cache can be reused."""
import argparse,json,subprocess,sys
from pathlib import Path
from _shared import PACKAGE,load_config,write_new

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--scans-file');p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);scans=a.scans_file or cfg['evaluation_split']
    print(json.dumps({'execute':a.execute,'output':str(a.output),'requires_GPU':True,
          'inputs':'RGB frames plus existing scan/siglip2_val/{patch_tokens_fp16.npy,paths.json}'}),flush=True)
    if a.execute:
        if a.output.exists():raise FileExistsError('Use a new output directory')
        resolved=a.output.with_name(a.output.name+'.config.json');write_new(resolved,cfg)
        subprocess.run([sys.executable,str(PACKAGE/'src/precompute_test_cache_shard.py'),'--config',str(resolved),
                        '--output',str(a.output),'--scans-file',str(scans)],check=True)
