import hashlib,importlib.util,shutil,tempfile,unittest
from pathlib import Path

PACKAGE=Path(__file__).resolve().parents[1]

def load(path):
    spec=importlib.util.spec_from_file_location('isolated_matching_compat',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class MatchingCompatTests(unittest.TestCase):
    def test_relocated_package_uses_its_own_legacy_parser(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'relocated';(root/'src').mkdir(parents=True);(root/'legacy_matching').mkdir()
            for rel in ['src/matching_compat.py','legacy_matching/eva_qwen8b.py']:
                shutil.copy2(PACKAGE/rel,root/rel)
            module=load(root/'src/matching_compat.py')
            self.assertEqual(module.LEGACY,root/'legacy_matching/eva_qwen8b.py')
            self.assertEqual(module.legacy_candidates({'match_pred_node_ids':10,'similarity_scores':'5'}),
                             (['10','none','none'],[5,1,1]))
            self.assertEqual(module.legacy_conflict({'best_matching_gt_obj_id':7})[0],'7')

    def test_unknown_ids_abstain_but_old_padding_is_preserved(self):
        module=load(PACKAGE/'src/matching_compat.py')
        ids,scores,note=module.candidate_record({'match_pred_node_ids':['10','missing'],'similarity_scores':[5,4]}, {'10'})
        self.assertEqual(ids,['10','none','none']);self.assertEqual(scores,[5,4,1])
        self.assertEqual(note['unknown_ids_abstained'],['missing'])

if __name__=='__main__':unittest.main()
