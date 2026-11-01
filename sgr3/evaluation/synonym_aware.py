"""Apply the SAME frozen equivalence relation to both endpoint labels for all methods."""
from collections import defaultdict
from copy import deepcopy
from .base import empty_counts
from .aliases import lexical

PROTOCOL = 'robio-synonym-aware-directed-set-v3'


def canonical(label, aliases):
    label = lexical(label)
    return aliases.get(label, label)


def score_record(strict, aliases):
    result = deepcopy(strict)
    result['strict_category_counts'] = dict(strict['counts'])
    result['counts'] = empty_counts()
    result['per_predicate'] = {}
    good = {}
    op = defaultdict(lambda: dict(total=0, matched=0))
    for d in result['object_class_details']:
        gt, pred = canonical(d['gt_class'], aliases), canonical(d['pred_class'], aliases)
        ok = d['pred_id'] is not None and bool(gt and pred and gt == pred)
        d.update(strict_class_correct=d['class_correct'], canonical_gt_class=gt,
                 canonical_pred_class=pred, class_correct=ok)
        d['class_match_kind'] = 'exact' if d['strict_class_correct'] else 'synonym' if ok else 'mismatch' if d['pred_id'] is not None else 'unmapped'
        good[d['gt_id']] = ok
        op[d['gt_class']]['total'] += 1
        op[d['gt_class']]['matched'] += int(ok)
    result['matched_objects'] = sum(good.values())
    result['object_per_class'] = dict(op)
    for d in result['details']:
        assert len(d['candidates']) <= 3
        both = good[d['src']] and good[d['tgt']]
        hit = bool(both and d['legacy_hit'])
        old = bool(both and d['candidates'])
        d.update(strict_category_hit=d['hit'], endpoint_classes_correct=both, hit=hit)
        c = result['per_predicate'].setdefault(d['predicate'], empty_counts())
        for counts in [result['counts'], c]:
            counts['hits'] += int(hit)
            counts['old'] += int(old)
            counts['both'] += int(both)
            counts['new'] += 1
    result['hits_with_legacy_old_denominator']['hits'] = result['counts']['hits']
    c, s, l = result['counts'], strict['counts'], strict['legacy_counts']
    assert c['hits'] <= c['old'] <= c['both'] <= c['new']
    assert c['new'] == s['new'] == l['new']
    for key in ['hits', 'old', 'both']:
        assert s[key] <= c[key] <= l[key]
    return result
