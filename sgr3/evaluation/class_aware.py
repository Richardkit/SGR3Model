"""Separate category-aware scoring; never mutates predictions or legacy scoring.

Fixed object-class equality after case/whitespace normalization. This is a
strict taxonomy check, not a semantic-equivalence judge for open names.
"""
from collections import defaultdict
from copy import deepcopy
from .base import score_scan as legacy_score, empty_counts, identity, validate_mapping

PROTOCOL = 'robio-class-aware-directed-set-v2'


def class_name(node):
    value = node.get('label') or node.get('name') or ''
    if not isinstance(value, str):
        raise ValueError('A node class must be a single explicit string')
    return ' '.join(value.strip().lower().split())


def score_scan(gt, pred, mapping, k=3):
    mapping = validate_mapping(gt, pred, mapping)
    legacy = legacy_score(gt, pred, mapping, k)
    result = deepcopy(legacy)
    result['legacy_counts'] = dict(legacy['counts'])
    result['counts'] = empty_counts()
    result['per_predicate'] = {}
    pn = {identity(n['id']): n for n in pred.get('nodes', [])}
    good = {}
    object_details = []
    op = defaultdict(lambda: dict(total=0, matched=0))
    for n in gt.get('nodes', []):
        gid = identity(n['id'])
        pid = mapping.get(gid)
        node = pn.get(str(pid)) if pid is not None else None
        target = class_name(n)
        output = class_name(node) if node is not None else ''
        good[gid] = bool(target and output and target == output)
        op[target]['total'] += 1
        op[target]['matched'] += int(good[gid])
        object_details.append(dict(gt_id=gid, pred_id=pid, gt_class=target,
                                   pred_class=output, class_correct=good[gid]))
    result['instance_associated_objects'] = legacy['matched_objects']
    result['matched_objects'] = sum(good.values())
    result['object_per_class'] = dict(op)
    result['object_class_details'] = object_details
    result['hits_with_legacy_old_denominator'] = dict(hits=0, old=legacy['counts']['old'])
    for detail in result['details']:
        both = good[detail['src']] and good[detail['tgt']]
        old = both and bool(detail['candidates'])
        hit = both and detail['hit']
        detail.update(legacy_hit=detail['hit'], endpoint_classes_correct=both, hit=bool(hit))
        per = result['per_predicate'].setdefault(detail['predicate'], empty_counts())
        for counts in [result['counts'], per]:
            counts['hits'] += int(hit)
            counts['old'] += int(old)
            counts['both'] += int(both)
            counts['new'] += 1
    result['hits_with_legacy_old_denominator']['hits'] = result['counts']['hits']
    c = result['counts']
    assert c['hits'] <= c['old'] <= c['both'] <= c['new']
    assert c['new'] == legacy['counts']['new']
    assert c['hits'] <= legacy['counts']['hits']
    return result
