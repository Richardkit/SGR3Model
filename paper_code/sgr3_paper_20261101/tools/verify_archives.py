"""CPU regression against the already frozen main/ablation results, no rematching."""
import argparse,json,sys,tempfile
from pathlib import Path
from _shared import PACKAGE,DEFAULT_PROJECT,write_new
sys.path.insert(0,str(PACKAGE))
from evaluation.evaluate import evaluate,digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=DEFAULT_PROJECT)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();root=a.project_root.resolve()
    audit=json.loads((root/'paper_revision/frozen_tables_20261031/FROZEN_AUDIT.json').read_text())
    checks={};sources={}
    for group,item in audit['results'].items():
        path=Path(item['result_source']);saved=json.loads(path.read_text());sources[str(path)]=digest(path)
        for source,expected in saved['source_sha256'].items():
            if source not in sources:sources[source]=digest(source)
            assert sources[source]==expected,source
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            if group.startswith('main/'):
                candidates=[Path(x) for x in saved['source_sha256'] if Path(x).parent.name=='manifests' or Path(x).name in ['manifest.json','evaluation_manifest.json']]
                assert len(candidates)==1;manifest=candidates[0]
            else:
                states={}
                for row in saved['per_scan']:
                    sid=row['scan'];mp=td/f'{sid}.json';mp.write_text(json.dumps(row['effective_mapping']))
                    states[sid]={'state':'complete','prediction':row['prediction'],'mapping':str(mp)}
                manifest=td/'manifest.json';manifest.write_text(json.dumps({'method':group,
                    'split':str(PACKAGE/'splits/evaluation_scans.txt'),'GT_root':str(root/'data/3RScan'),'scans':states}))
            actual=evaluate(manifest)
        for key in ['counts','recall','mRecall_by_denominator','per_predicate']:assert actual[key]==saved[key],(group,key)
        assert actual['complete_scans']==47 and actual['counts']['new']==10125
        expected={r['scan']:r for r in saved['per_scan']}
        for row in actual['per_scan']:
            old=expected[row['scan']]
            assert row['counts']==old['counts'] and row['matched_objects']==old['matched_objects']
            for newd,oldd in zip(row['details'],old['details']):
                for key in ['src','tgt','pred_src','pred_tgt','predicate','candidates','hit']:assert newd[key]==oldd[key]
        checks[group]={'passed':True,'scans':47,'counts':actual['counts'],
                       'object_category_recall':actual['object_category_recall_after_association'],
                       'New_mRecall':actual['mRecall_by_denominator']['new'],'node_ratio':actual['node_ratio']}
        print(group,'PASS',flush=True)
    write_new(a.output,{'all_passed':True,'groups':len(checks),'scan_cases':sum(c['scans'] for c in checks.values()),
                       'checks':checks,'source_sha256':sources,
                       'evaluation_source_sha256':{str(x):digest(x) for x in (PACKAGE/'evaluation').glob('*') if x.is_file()}})

if __name__=='__main__':main()
