"""Dependency-free corrected scene-graph SET recall. Never native joint R@1."""
from __future__ import annotations
import ast
from collections import Counter, defaultdict

PROTOCOL = 'robio-directed-set-v1'
ALIASES = {'in front of':'front','infront of':'front','infront':'front',
           'behind of':'behind','next to':'close by','near':'close by'}
NEGATIVE = {'none','no relation','__none__'}

def normalize(value):
    value = str(value).strip().lower() if value is not None else ''
    return ALIASES.get(value, value)

def labels(value):
    if value is None:return []
    if isinstance(value,str):
        try: parsed=ast.literal_eval(value)
        except (ValueError,SyntaxError):parsed=None
        value=parsed if isinstance(parsed,list) else value.split(',')
    if not isinstance(value,list):value=[value]
    return [normalize(v) for v in value if normalize(v) and normalize(v) not in NEGATIVE]

def identity(value):
    if value is None:return None
    value=str(value).strip()
    return None if not value or value.lower()=='none' else value

def node_ids(graph):
    ids=[identity(n.get('id')) for n in graph.get('nodes',[])]
    if None in ids or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate object ID')
    return set(ids)

def validate_mapping(gt,pred,mapping):
    gids,pids=node_ids(gt),node_ids(pred)
    if set(map(str,mapping))-gids:raise ValueError('Mapping refers to unknown GT ID')
    result={g:identity(mapping.get(g)) for g in gids}
    values=[p for p in result.values() if p is not None]
    if set(values)-pids:raise ValueError('Mapping refers to unknown predicted ID')
    if len(values)!=len(set(values)):raise ValueError('Object mapping must be one-to-one')
    return result

def candidate_pairs(pred,k=3):
    """Union duplicate directed rows FIRST; normalize/deduplicate BEFORE Top-K."""
    if not isinstance(k,int) or k<1:raise ValueError('Positive explicit candidate budget required')
    ids=node_ids(pred);pairs={};raw=Counter();invalid=0
    for e in pred.get('edges',[]):
        s,t=identity(e.get('src')),identity(e.get('tgt'))
        if s not in ids or t not in ids or s==t:
            invalid+=1;continue
        key=(s,t);values=pairs.setdefault(key,[])
        value=e.get('relationship')
        if value is None:value=e.get('c_edge')
        for rel in labels(value):
            if rel not in values:values.append(rel)
        raw[key]=len(values)
    pairs={p:v[:k] for p,v in pairs.items() if v}
    return pairs,{'raw_pairs_over_budget':sum(n>k for n in raw.values()),
                  'invalid_prediction_edges':invalid,'output_pairs':len(pairs),
                  'output_hypotheses':sum(map(len,pairs.values()))}

def empty_counts():return dict(hits=0,old=0,both=0,new=0)

def score_scan(gt,pred,mapping,k=3):
    """Each valid positive GT relationship RECORD counts, including multiple labels per pair."""
    mapping=validate_mapping(gt,pred,mapping);gids=node_ids(gt)
    pairs,diagnostics=candidate_pairs(pred,k);counts=empty_counts();per={};details=[];invalid_gt=0
    object_per=defaultdict(lambda:dict(total=0,matched=0))
    for n in gt.get('nodes',[]):
        label=normalize(n.get('label',n.get('name','unknown')))
        object_per[label]['total']+=1
        object_per[label]['matched']+=int(mapping[str(n['id'])] is not None)
    for e in gt.get('edges',[]):
        s,t=identity(e.get('src')),identity(e.get('tgt'))
        value=e.get('relationship')
        if value is None:value=e.get('predicate')
        # Native GT schema contains one positive predicate per record. Do not silently
        # reinterpret list-valued annotations as a new counting protocol.
        if isinstance(value,list):raise ValueError('GT requires one predicate per relationship record')
        rel=normalize(value)
        if s not in gids or t not in gids or s==t or not rel or rel in NEGATIVE:
            invalid_gt+=1;continue
        ps,pt=mapping[s],mapping[t];both=ps is not None and pt is not None
        candidates=pairs.get((ps,pt),[]) if both else []
        hit=int(rel in candidates);old=int(bool(candidates));c=per.setdefault(rel,empty_counts())
        for target in [counts,c]:
            target['new']+=1;target['both']+=int(both);target['old']+=old;target['hits']+=hit
        details.append(dict(src=s,tgt=t,predicate=rel,pred_src=ps,pred_tgt=pt,
                            candidates=candidates,hit=bool(hit)))
    assert counts['hits']<=counts['old']<=counts['both']<=counts['new']
    return {'counts':counts,'per_predicate':per,'GT_objects':len(gids),
            'matched_objects':sum(v is not None for v in mapping.values()),
            'object_per_class':dict(object_per),'invalid_GT_records':invalid_gt,
            'diagnostics':diagnostics,'details':details}

def aggregate(rows,k=3):
    totals=empty_counts();per={};objects=Counter();op=defaultdict(Counter);diag=Counter()
    for r in rows:
        for key,value in r['counts'].items():totals[key]+=value
        for label,c in r['per_predicate'].items():
            p=per.setdefault(label,empty_counts())
            for key,value in c.items():p[key]+=value
        objects.update({key:r[key] for key in ['GT_objects','matched_objects']})
        for label,c in r['object_per_class'].items():op[label].update(c)
        diag.update(r['diagnostics'])
    ratios={name:totals['hits']/totals[name] if totals[name] else 0.0 for name in ['old','both','new']}
    macro={};supports={}
    for scope in ['old','both','new']:
        supported=[c for c in per.values() if c[scope]>0];supports[scope]=len(supported)
        macro[scope]=sum(c['hits']/c[scope] for c in supported)/len(supported) if supported else 0.0
    return {'protocol':PROTOCOL,'metric_name':f'RelationshipSetRecall@{k}',
            'candidate_budget':k,'ranking':'normalized distinct exported order; not calibrated joint probability',
            'semantic_judge':'OFF','counts':totals,'recall':ratios,
            'mRecall_by_denominator':macro,'mRecall_supported_class_counts':supports,
            'per_predicate':per,'objects':dict(objects),
            'object_instance_recall':objects['matched_objects']/objects['GT_objects'] if objects['GT_objects'] else 0.0,
            'object_instance_mRecall':sum(c['matched']/c['total'] for c in op.values())/len(op) if op else 0.0,
            'diagnostics':dict(diag)}

def scene_groups(metadata):
    mapping={}
    for group in metadata:
        ref=group['reference']
        for sid in [ref]+[s['reference'] for s in group.get('scans',[])]:
            if sid in mapping:raise ValueError('Duplicate scan in official scene metadata')
            mapping[sid]=ref
    return mapping

def check_kb_members(members,test_ids,metadata):
    groups=scene_groups(metadata);members=set(members);test_ids=set(test_ids)
    if (members|test_ids)-groups.keys():raise ValueError('Unknown scan in KB or test split')
    exact=sorted(members&test_ids);test_groups={groups[s] for s in test_ids}
    rescans=sorted(s for s in members if groups[s] in test_groups)
    return {'passed':not rescans,'exact_scan_overlap':exact,'same_scene_overlap':rescans}

def image_inputs(original,kept,window,warmup=2,policy='original_window'):
    if not original or len(original)!=len(set(original)):raise ValueError('Invalid original image list')
    if len(kept)!=len(set(kept)) or not set(kept)<=set(original):raise ValueError('Invalid retained image list')
    effective=list(original) if window<warmup else list(kept)
    if not effective:return {'query_images':[],'generation_images':[],'skip':True}
    if policy not in ['original_window','filtered']:raise ValueError('Unknown image policy')
    return {'query_images':effective,'generation_images':list(original) if policy=='original_window' else effective,'skip':False}

def validate_judge_result(result,candidates):
    """Optional legacy replay only: a judge may only support supplied candidates."""
    supported=labels(result.get('supported_predicates',[]));allowed=set(labels(candidates))
    if set(supported)-allowed:raise ValueError('Judge invented an unsupplied predicate')
    return supported
