"""Fixed category-equivalent set@3 scoring on an explicit complete test list."""
import argparse,hashlib,json
from pathlib import Path
from .base import aggregate
from .class_aware import score_scan
from .synonym_aware import score_record

POLICY=Path(__file__).with_name('object_aliases_v2.json')
PROTOCOL='robio-category-equivalence-directed-set-v4'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def resolve(base,value):
    p=Path(value);return p if p.is_absolute() else base/p

def evaluate(manifest_path):
    path=Path(manifest_path).resolve();base=path.parent;m=read(path)
    split=resolve(base,m['split']);ids=split.read_text().split()
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Empty or duplicate scan list')
    if set(m['scans'])-set(ids):raise ValueError('Manifest contains scans outside the test split')
    aliases=read(POLICY)['alias_to_canonical'];rows=[];states={};predicted_nodes=0;instance_hits=0
    sources={str(path):digest(path),str(split):digest(split),str(POLICY):digest(POLICY)}
    for sid in ids:
        spec=m['scans'].get(sid,{})
        gtpath=resolve(base,spec.get('gt',str(Path(m['GT_root'])/sid/f'{sid}_scan_sg.json')))
        gt=read(gtpath);sources[str(gtpath)]=digest(gtpath)
        state=spec.get('state','pending')
        if state not in {'complete','failed','pending','not_available'}:raise ValueError('Unknown scan state')
        pred={'nodes':[],'edges':[]};mapping={}
        if state=='complete':
            pp=resolve(base,spec['prediction']);mp=resolve(base,spec['mapping'])
            if not pp.is_file() or not mp.is_file():state='artifact_missing'
            else:
                pred=read(pp);mapping=read(mp);mapping=mapping.get('gt2pred',mapping)
                sources.update({str(pp):digest(pp),str(mp):digest(mp)})
        row=score_record(score_scan(gt,pred,mapping,3),aliases)
        row.update(scan=sid,state=state)
        rows.append(row);states[sid]=state
        predicted_nodes+=len(pred['nodes']);instance_hits+=row['instance_associated_objects']
    result=aggregate(rows,3)
    result['object_category_recall_after_association']=result.pop('object_instance_recall')
    result['object_category_mRecall_after_association']=result.pop('object_instance_mRecall')
    result['objects']['correct_class_objects']=result['objects'].pop('matched_objects')
    result['objects'].update(instance_associated_objects=instance_hits,predicted_nodes=predicted_nodes)
    result['node_ratio']=predicted_nodes/result['objects']['GT_objects'] if result['objects']['GT_objects'] else 0
    pending=[s for s,t in states.items() if t in {'pending','artifact_missing','not_available'}]
    failed=[s for s,t in states.items() if t=='failed']
    result.update(protocol=PROTOCOL,metric_name='CategoryEquivalentRelationshipSetRecall@3',method=m['method'],
                  planned_scans=len(ids),complete_scans=sum(s=='complete' for s in states.values()),
                  pending_scans=pending,failed_scans=failed,
                  final=not pending,all_scans_successful=not pending and not failed,
                  scope='All fixed-list positive GT records; missing/failed predictions retain GT and contribute zero hits. Pending results are not final.',
                  object_metric_note='Category-correct associated objects / all GT objects; NOT classifier R@10 or instance-only matching recall.',
                  alias_table_sha256=digest(POLICY),semantic_judge='OFF',
                  source_sha256=sources,per_scan=rows,
                  evaluator_sha256={str(p):digest(p) for p in Path(__file__).parent.glob('*.py')})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use a new output path')
    result=evaluate(a.manifest);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['method','complete_scans','planned_scans','final','counts','recall','mRecall_by_denominator','object_category_recall_after_association','node_ratio']},ensure_ascii=False))

if __name__=='__main__':main()
