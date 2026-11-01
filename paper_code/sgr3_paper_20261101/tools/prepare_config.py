"""Resolve local paths and validate the screenshot's configuration. Does not run models."""
import argparse,json
from _shared import load_config,write_new

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--project-root');p.add_argument('--output',required=True)
    a=p.parse_args();cfg=load_config(a.config,a.project_root);write_new(a.output,cfg)
    print(json.dumps({'config':a.output,'patch_topk':10,'reference_topk':5,'scene_exclusion':True,'model_loaded':False}))

