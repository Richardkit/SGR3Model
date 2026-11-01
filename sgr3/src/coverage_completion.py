"""One bounded image-grounded completion call for missing visible object pairs."""
from copy import deepcopy
import json

from common import decode_predicate_ids, validate_delta, write_json
from dashscope_client import parse_json
from soft_output import parse_model_output


def missing_pairs(previous, delta):
    nodes = {str(n['id']): n for n in previous['nodes'] + delta['new_nodes']}
    visible = {str(n['id']) for n in delta['new_nodes']}
    visible.update(str(i) for i in delta.get('seen_old_nodes', []))
    visible.update(str(e[k]) for e in delta['new_edges'] for k in ['src', 'tgt'])
    if not visible <= nodes.keys():
        raise ValueError('Unknown visible node')
    existing = {(str(e['src']), str(e['tgt'])) for e in previous['edges'] + delta['new_edges']}
    covered = sum(s in visible and t in visible for s, t in existing)
    target = len(visible) * min(3, max(0, len(visible) - 1))
    pairs = [(s, t) for s in sorted(visible, key=int) for t in sorted(visible, key=int)
             if s != t and (s, t) not in existing]
    return nodes, visible, pairs if covered < target else []


def merge_completion(delta, extra, requested, already_present=frozenset()):
    result = deepcopy(delta)
    seen = set()
    for edge in extra['new_edges']:
        pair = (str(edge['src']), str(edge['tgt']))
        if pair in already_present:
            continue  # Recorded separately; never overwrite an existing ranked proposal.
        if pair not in requested or pair in seen:
            result.setdefault('_soft_events', []).append({'action':'ignore_unrequested_or_duplicate_completion_pair','raw':edge})
            continue
        seen.add(pair)
        result['new_edges'].append(edge)
    return result


def complete_pairs(client, messages, previous, delta, classes, predicates, encoding, prefix):
    nodes, visible, pairs = missing_pairs(previous, delta)
    if not pairs:
        return delta, 0
    if len(pairs) > 128:
        write_json(str(prefix) + '.completion.skipped.json', {'reason':'Original128-pair completion budget exceeded; primary prediction preserved','missing_pairs':len(pairs)})
        return delta, 0
    instruction = f'''Perform a relation coverage completion using the SAME images, vocabulary and reference evidence.
These are the model's own visible object IDs and descriptions, not ground truth:
{json.dumps([nodes[i] for i in sorted(visible, key=int)], ensure_ascii=False)}
Systematically check EVERY ordered candidate pair listed below:
{json.dumps(pairs)}
The existing graph has too few relations. Check spatial relations as well as physical support/attachment.
For EACH candidate pair supported by the images, output its src, tgt and a ranked list of distinct
allowed predicates, with the most likely first. Keep the existing {encoding} predicate encoding.
Omit unsupported pairs; do not invent objects or relations. Inspect both directions separately.
Do not return existing pairs, new objects, an example, or a short summary.
Return JSON only: {{"new_edges":[{{"src":1,"tgt":2,"c_edge":[1,2,3]}}]}}.
The example is syntax only; choose actual IDs and predicates from the supplied candidates and images.'''
    request = deepcopy(messages)
    request.append({'role': 'user', 'content': [{'type': 'text', 'text': instruction}]})
    write_json(str(prefix) + '.completion.messages.json', request)
    write_json(str(prefix) + '.completion.pairs.json', pairs)
    response = client.complete(request)
    write_json(str(prefix) + '.completion.response.json', response)
    raw, recovery_notes = parse_model_output(response, main=False)
    write_json(str(prefix) + '.completion.recovery.json', recovery_notes)
    write_json(str(prefix) + '.completion.raw.json', raw)
    extra = decode_predicate_ids(raw, predicates) if encoding == 'ids' else raw
    existing = {(str(e['src']), str(e['tgt'])) for e in previous['edges'] + delta['new_edges']}
    ignored = [e for e in extra['new_edges'] if (str(e['src']), str(e['tgt'])) in existing]
    write_json(str(prefix) + '.completion.ignored_existing.json', ignored)
    combined = merge_completion(delta, extra, set(pairs), existing)
    validate_delta(combined, previous, classes, predicates)
    return combined, 1
