"""Read-only official scene-family audit; optionally inspect all actual index keys."""
import argparse,json,pickle,re,sys
from pathlib import Path
from _shared import load_config,PACKAGE,write_new
sys.path.insert(0,str(PACKAGE))
from evaluation.base import check_kb_members
from evaluation.evaluate import digest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--project-root')
    p.add_argument('--actual-keys',action='store_true');p.add_argument('--output')
    a=p.parse_args();cfg=load_config(a.config,a.project_root);kb=Path(cfg['kb_dir'])
    manifest=json.loads((kb/'manifest.json').read_text());ids=Path(cfg['evaluation_split']).read_text().split()
    meta=json.loads(Path(cfg['scene_metadata']).read_text())
    assert digest(cfg['evaluation_split'])==manifest['split_sha256']
    assert digest(cfg['scene_metadata'])==manifest['metadata_sha256']
    result=check_kb_members(manifest['included_scans'],ids,meta)
    result.update(scope='manifest membership',index_built=False)
    if a.actual_keys:
        with (kb/'keys.pkl').open('rb') as f:keys=pickle.load(f)
        members=set()
        for k in keys:
            k=k.decode() if isinstance(k,bytes) else str(k)
            match=re.search(r'[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}',k)
            if not match:raise ValueError('Malformed KB key')
            members.add(match[0].lower())
        if members!=set(manifest['included_scans']):raise ValueError('Key membership disagrees with KB manifest')
        result=check_kb_members(members,ids,meta)
        result.update(scope='actual key membership',keys=len(keys),keys_sha256=digest(kb/'keys.pkl'),index_built=False)
    if a.output:write_new(a.output,result)
    print(json.dumps(result,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)
