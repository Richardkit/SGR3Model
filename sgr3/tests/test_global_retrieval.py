"""Exercise the actual retrieval function without importing/loading GPU models."""
import ast,json,os,re,sys,tempfile,unittest
from collections import defaultdict,Counter
from copy import deepcopy
from pathlib import Path
from typing import List,Dict,Any,Tuple,Optional
import numpy as np
P=Path(__file__).resolve().parents[1]

class GlobalRetrievalTests(unittest.TestCase):
    def test_frames_from_multiple_scans_compete_by_cumulative_score(self):
        source=P/'src/vision_legacy.py'
        names={'parse_kb_key_to_scan_frame','_merge_reference_sgs','merge_reference_scenes','run_rag_for_window'}
        functions=[x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names]
        ns=dict(np=np,os=os,re=re,json=json,defaultdict=defaultdict,Counter=Counter,deepcopy=deepcopy,
                List=List,Dict=Dict,Any=Any,Tuple=Tuple,Optional=Optional,RETRIEVAL_TEMPERATURE=.5,
                UNWEIGHTED=False,UUID_RE=re.compile(r'[0-9a-f-]{36}'),
                atomic_write_json=lambda p,d:Path(p).write_text(json.dumps(d)))
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
        A,B,Q=[str(i)*8+'-'+str(i)*4+'-'+str(i)*4+'-'+str(i)*4+'-'+str(i)*12 for i in [1,2,3]]
        class Index:
            count=0
            def search(self,query,topk):
                assert topk==10
                first=self.count;self.count+=1
                scores=np.full((2,10),-1e9,np.float32);indices=np.full((2,10),-1,np.int64)
                scores[:,:2]=[.9,.6];indices[:,:2]=[first,2]
                return scores,indices
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for sid,fid in [(A,1),(A,2),(B,3)]:
                p=root/sid/'frame_sg'/f'frame-{fid:06d}.json';p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps({'objects':[]}))
            output=root/'result.json';images=['q0','q1']
            ns['run_rag_for_window'](str(root/Q/'sequence'),Index(),
                [f'{A}/frame-000001',f'{A}/frame-000002',f'{B}/frame-000003'],str(root),images,None,str(output),
                patch_topk=10,faiss_topk=2,excluded_scan_ids={Q},known_scan_ids={A,B,Q},
                query_patch_cache={i:np.eye(2,dtype=np.float32) for i in images},keys_are_normalized=True)
            report=json.loads(output.read_text())
            self.assertEqual(report['selection_policy'],'global_frame_sum_topk_v1')
            frames=report['top_reference_frames']
            self.assertEqual(frames[0]['scan_id'],B)  # .6+.6 beats either .9 frame.
            self.assertEqual({x['scan_id'] for x in frames},{A,B})
            self.assertEqual(len(report['loaded_reference_frames']),2)

if __name__=='__main__':unittest.main()
