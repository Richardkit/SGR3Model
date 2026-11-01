import json,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(P),str(P/'src'),str(P/'tools')]
from evaluation.base import check_kb_members,image_inputs
from evaluation.class_aware import score_scan
from evaluation.synonym_aware import score_record
from evaluation.evaluate import evaluate
from retrieval_cache import signature
from _shared import load_config
ALIASES=json.loads((P/'evaluation/object_aliases_v2.json').read_text())['alias_to_canonical']

class ProtocolTests(unittest.TestCase):
    def test_scene_family_excludes_unrelated_uuid_prefix(self):
        meta=[{'reference':'aaa','scans':[{'reference':'zzz'}]},{'reference':'aab','scans':[]}]
        self.assertFalse(check_kb_members(['zzz'],['aaa'],meta)['passed'])
        self.assertTrue(check_kb_members(['aab'],['aaa'],meta)['passed'])

    def test_original_window_and_first_two_warmup(self):
        original=['a','b','c']
        for wi in [0,1]:self.assertEqual(image_inputs(original,[],wi)['generation_images'],original)
        self.assertEqual(image_inputs(original,['a'],2)['generation_images'],original)
        self.assertTrue(image_inputs(original,[],2)['skip'])

    def test_wrong_cache_policy_or_k_is_not_reused(self):
        cfg=load_config();old=dict(cfg,patch_topk=5)
        self.assertNotEqual(signature(cfg),signature(old))
        with self.assertRaises(ValueError):signature(dict(cfg,retrieval_selection_policy='scene_argmax_v1'))

    def graphs(self):
        gt={'nodes':[{'id':1,'label':'chair'},{'id':2,'label':'table'}],
            'edges':[{'src':1,'tgt':2,'relationship':'left'},{'src':1,'tgt':2,'relationship':'right'}]}
        pred={'nodes':[{'id':10,'name':'armchair'},{'id':20,'name':'desk'}],
              'edges':[{'src':10,'tgt':20,'c_edge':['left','left','front']},
                       {'src':10,'tgt':20,'c_edge':['behind','right']}]}
        return gt,pred,{'1':'10','2':'20'}

    def test_aliases_both_endpoints_and_three_distinct_candidates(self):
        gt,pred,m=self.graphs();row=score_record(score_scan(gt,pred,m,3),ALIASES)
        self.assertEqual(row['counts'],{'hits':1,'old':2,'both':2,'new':2})
        self.assertEqual(row['matched_objects'],2)
        pred['nodes'][1]['name']='sofa';row=score_record(score_scan(gt,pred,m,3),ALIASES)
        self.assertEqual(row['counts'],{'hits':0,'old':0,'both':0,'new':2})

    def test_mapping_must_be_one_to_one(self):
        gt,pred,m=self.graphs()
        with self.assertRaises(ValueError):score_scan(gt,pred,{'1':'10','2':'10'},3)

    def test_full_denominator_zero_hit_classes_and_pending_scan(self):
        gt,pred,m=self.graphs()
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'split.txt').write_text('a\nb\n')
            for sid in ['a','b']:
                (d/sid).mkdir();(d/sid/f'{sid}_scan_sg.json').write_text(json.dumps(gt))
            (d/'pred.json').write_text(json.dumps(pred));(d/'map.json').write_text(json.dumps(m))
            (d/'manifest.json').write_text(json.dumps({'method':'fixture','split':'split.txt','GT_root':str(d),
                'scans':{'a':{'state':'complete','prediction':'pred.json','mapping':'map.json'}}}))
            result=evaluate(d/'manifest.json')
            self.assertFalse(result['final']);self.assertEqual(result['counts']['new'],4)
            self.assertEqual(result['recall']['new'],.25)
            self.assertEqual(result['mRecall_by_denominator']['new'],.25)
            self.assertEqual(result['object_category_recall_after_association'],.5)

if __name__=='__main__':unittest.main()
