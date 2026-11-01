"""Legacy candidate decoding; unusable evidence abstains per object, never invents IDs."""
import ast
import hashlib
import json
from pathlib import Path

LEGACY=Path(__file__).resolve().parents[1]/'legacy_matching/eva_qwen8b.py'
source=LEGACY.read_text(encoding='utf-8')
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_qwen_match_inference')
PARSER_SHA256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()
# The original no_grad decorator concerns local inference; decoding itself has no tensors.
node.decorator_list=[]


def legacy_candidates(data):
    # Execute the actual original function, replacing only its local model call.
    namespace={'json':json,'run_qwen_vllm':lambda *a,**kw:(data,json.dumps(data,ensure_ascii=False))}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(LEGACY),'exec'),namespace)
    ids,scores,_=namespace['run_qwen_match_inference'](None,None)
    return ids,scores


conflict_node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_qwen_pred_vs_gt_inference')
CONFLICT_PARSER_SHA256=hashlib.sha256(ast.get_source_segment(source,conflict_node).encode()).hexdigest()
conflict_node.decorator_list=[]

def legacy_conflict(data):
    namespace={'json':json,'run_qwen_vllm':lambda *a,**kw:(data,json.dumps(data,ensure_ascii=False))}
    exec(compile(ast.Module(body=[conflict_node],type_ignores=[]),str(LEGACY),'exec'),namespace)
    selected,raw=namespace['run_qwen_pred_vs_gt_inference'](None,None)
    return selected,raw


def candidate_record(data,candidate_ids):
    ids,scores=legacy_candidates(data)
    unknown=[x for x in ids if x!='none' and x not in candidate_ids]
    safe=[x if x=='none' or x in candidate_ids else 'none' for x in ids]
    real=[x for x in safe if x!='none']
    note={'legacy_parser_sha256':PARSER_SHA256,'unknown_ids_abstained':unknown,
          'duplicate_ids_retained_as_legacy':len(real)!=len(set(real)),
          'unequal_arrays_ignored_by_legacy_resolver':len(safe)!=len(scores),
          'decoded_ids':safe,'decoded_scores':scores}
    return safe,scores,note


def review_record(data):
    if isinstance(data,dict) and type(data.get('same_physical_instance')) is bool:
        return dict(data,reason=data.get('reason')),None
    return {'same_physical_instance':False,'reason':None},'Unusable review; no supported acceptance; correspondence left unmatched'
