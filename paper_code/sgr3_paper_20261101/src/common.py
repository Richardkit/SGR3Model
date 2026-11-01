import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def scene_map(metadata):
    result = {}
    for group in metadata:
        ref = group['reference']
        for sid in [ref] + [s['reference'] for s in group['scans']]:
            if sid in result:
                raise ValueError('Repeated scan in scene metadata: ' + sid)
            result[sid] = ref
    return result


def excluded_scans(metadata, evaluation_ids):
    mapping = scene_map(metadata)
    missing = set(evaluation_ids) - mapping.keys()
    if missing:
        raise ValueError('Evaluation IDs missing from scene metadata: ' + repr(sorted(missing)))
    refs = {mapping[s] for s in evaluation_ids}
    return {s for s, ref in mapping.items() if ref in refs}



def retrieval_exclusions(metadata, evaluation_ids, policy='scene'):
    evaluation_ids = set(evaluation_ids)
    scene_ids = excluded_scans(metadata, evaluation_ids)
    if policy == 'scene':
        return scene_ids
    if policy == 'exact_scan':
        return evaluation_ids
    raise ValueError('Unknown retrieval exclusion policy: ' + str(policy))


def select_generation_images(original, kept, image_policy='filtered'):
    if (not original or len(set(original)) != len(original) or not kept
            or len(set(kept)) != len(kept) or not set(kept) <= set(original)):
        raise ValueError('Generation requires a nonempty, unique kept subset of window images')
    if image_policy == 'filtered':
        return list(kept)
    if image_policy == 'original_window':
        return list(original)
    raise ValueError('Unknown generation image policy: ' + str(image_policy))


def filtered_messages(original, kept, previous, references, classes, predicates, predicate_encoding='labels',
                      image_policy='filtered', coverage_policy='legacy'):
    from graph_legacy import build_prompt_window_rag
    images = select_generation_images(original, kept, image_policy)
    messages = build_prompt_window_rag(images, previous, references, classes, predicates,
                                     predicate_encoding=predicate_encoding)
    if coverage_policy == 'visible_pairs_v1':
        messages[0]['content'].append({'type': 'text', 'text': '''FINAL COVERAGE PASS (apply before returning JSON):
Inspect every supplied image, including background regions, for distinct visible objects.
Do not omit an independently visible movable object merely because you already mention
it in a furniture description. Keep structural surfaces that are visible. Do not split
an object into component parts or create invisible objects. Use only allowed class labels.

Make a private inventory of the N distinct objects visible in this window, including
reused IDs from PREV_GRAPH. Systematically examine EVERY ordered pair of different
visible objects, not just pairs involving newly added nodes. Include all visually
supported pairs, including spatial relations between distant visible objects.
For each source object, explicitly examine its min(3,N-1) closest distinct neighbours
and check the reverse directions too. Do not stop after a few support-to-floor edges.
The existing density request means at least N*min(3,N-1) supported directed pairs:
N=5 -> 15, N=6 -> 18, N=7 -> 21, N=8 -> 24. Recheck omissions if below this target;
never invent an object or unsupported relation merely to meet a count.

Include newly observed or reconfirmed old-to-old relations as well as relations
involving new nodes. Preserve old IDs; add only genuinely new objects to new_nodes.
Do not copy old relations that have no support in the current images.
For EVERY emitted pair choose a ranked list of distinct allowed predicates in descending
confidence order, the most likely first. Keep the required predicate encoding and
the existing JSON schema. Do not output your checklist or numerical counts.
Return the complete supported relation delta, not a short illustrative sample.'''.strip()})
    elif coverage_policy != 'legacy':
        raise ValueError('Unknown generation coverage policy')
    messages[0]['content'].append({'type':'text','text':'Predicate arrays are ranked, most likely first. Additional supported hypotheses are permitted; there is no hard generation limit of three. Keep the declared predicate encoding. Evaluation applies its own ranking and Top-K rules.'})
    actual = [c['image'] for m in messages for c in m['content'] if c['type'] == 'image']
    if actual != images:
        raise RuntimeError('Generation image list differs from the configured image policy')
    return messages


def decode_predicate_ids(raw_delta, predicates):
    """Keep every ranked hypothesis; decode IDs without inferring synonym labels."""
    from copy import deepcopy
    import ast
    delta = deepcopy(raw_delta)
    if not isinstance(delta, dict) or not isinstance(delta.get('new_edges'), list):
        raise ValueError('Expected delta with new_edges')
    decoded = []
    for edge in delta['new_edges']:
        if not isinstance(edge, dict):
            delta.setdefault('_soft_events', []).append({'action':'skip_nonobject_edge','raw':edge})
            continue
        values = edge.get('c_edge', edge.get('predicate', edge.get('relation', [])))
        if isinstance(values, str):
            try: parsed = ast.literal_eval(values)
            except (ValueError, SyntaxError): parsed = values
            values = parsed if isinstance(parsed, list) else [values]
        if not isinstance(values, list): values = [values] if values is not None else []
        labels = []
        for value in values:
            if type(value) is int or (isinstance(value,str) and value.strip().isdigit()):
                idx = int(value)
                label = predicates[idx-1] if 1 <= idx <= len(predicates) else '__unmapped_predicate_id_' + str(idx) + '__'
            elif isinstance(value,str):
                label = ' '.join(value.strip().lower().split())
            else:
                label = '__invalid_predicate_' + json.dumps(value,sort_keys=True,separators=(',',':')) + '__'
            if label and label not in labels: labels.append(label)
        if not labels:
            delta.setdefault('_soft_events', []).append({'action':'predicate_abstention','pair':[str(edge.get('src')),str(edge.get('tgt'))]})
            continue
        edge['c_edge'] = labels
        decoded.append(edge)
    delta['new_edges'] = decoded
    return delta


def normalize_delta(raw_delta, previous, predicates):
    delta = decode_predicate_ids(raw_delta, predicates)
    old = {str(n['id']) for n in previous['nodes']}
    seen = set();nodes=[];edges=[];pairs={}
    max_old=max((int(s) for s in old),default=0)
    for node in delta.get('new_nodes', []):
        if not isinstance(node,dict):
            delta.setdefault('_soft_events',[]).append({'action':'skip_nonobject_node','raw':node});continue
        sid=str(node.get('id',''))
        if (not sid.isdigit() or str(int(sid))!=sid or int(sid)<=max_old or sid in seen):
            delta.setdefault('_soft_events',[]).append({'action':'skip_repeated_or_invalid_node_id','raw':node});continue
        name=node.get('name','')
        if not isinstance(name,str) or not name.strip():name='__unknown_object__'
        desc=node.get('description','')
        if not isinstance(desc,str):desc=json.dumps(desc,ensure_ascii=False)
        node.update(name=name,description=desc);seen.add(sid);nodes.append(node)
    for edge in delta['new_edges']:
        pair=(str(edge.get('src','')),str(edge.get('tgt','')))
        if pair[0]==pair[1] or not set(pair)<=(old|seen):
            delta.setdefault('_soft_events',[]).append({'action':'skip_self_or_dangling_edge','raw':edge});continue
        if pair in pairs:
            existing=pairs[pair];existing['c_edge']=list(dict.fromkeys(existing['c_edge']+edge['c_edge']))
            delta.setdefault('_soft_events',[]).append({'action':'merge_duplicate_pair','pair':list(pair)});continue
        pairs[pair]=edge;edges.append(edge)
    if 'seen_old_nodes' in delta:
        raw_seen=delta['seen_old_nodes']
        raw_seen=raw_seen if isinstance(raw_seen,list) else []
        kept=[v for v in raw_seen if str(v) in old|seen]
        if kept!=raw_seen:delta.setdefault('_soft_events',[]).append({'action':'ignore_unknown_seen_ids','raw':raw_seen})
        delta['seen_old_nodes']=kept
    delta['new_nodes']=nodes;delta['new_edges']=edges
    return delta


def validate_delta(obj, previous, classes, predicates):
    # Do not silently salvage malformed model output or vote on copied global graphs.
    if not isinstance(obj, dict) or not {'new_nodes', 'new_edges'} <= obj.keys():
        raise ValueError('Expected new_nodes/new_edges delta JSON')
    nodes, edges = obj['new_nodes'], obj['new_edges']
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError('Nodes and edges must be lists')
    old = {str(n['id']): n for n in previous['nodes']}
    max_old = max((int(s) for s in old), default=0)
    new = set()
    for n in nodes:
        sid = str(n['id'])
        if not sid.isdigit() or str(int(sid)) != sid or int(sid) <= max_old or sid in new:
            raise ValueError('Invalid or reused new node ID')
        if not isinstance(n.get('name'), str) or not n['name'].strip() or not isinstance(n.get('description', ''), str):
            raise ValueError('Invalid node label/description')
        new.add(sid)
    needed = set()
    pairs = set()
    for e in edges:
        s, t = str(e['src']), str(e['tgt'])
        if s == t or (s, t) in pairs or not {s, t} <= (new | old.keys()):
            raise ValueError('Invalid, repeated or dangling edge')
        pairs.add((s, t))
        p = e.get('c_edge')
        if not isinstance(p, list) or not p or len(set(p)) != len(p) or any(not isinstance(v,str) or not v for v in p):
            raise ValueError('Expected nonempty distinct predicate labels')
        needed.update({s, t} & old.keys())
    stubs = [dict(old[s], description='') for s in sorted(needed)]
    # Keep the legacy merger from remapping IDs when the response orders new
    # nodes differently; votes and graph endpoints must share exactly the same IDs.
    return {'nodes': stubs + sorted(nodes, key=lambda n: int(n['id'])), 'edges': edges}


def add_votes(votes, edges):
    for edge in edges:
        key = (str(edge['src']), str(edge['tgt']))
        counters = votes.setdefault(key, {})
        for rank, predicate in enumerate(dict.fromkeys(edge['c_edge'])):
            count = counters.setdefault(predicate, [0, 0])
            count[0] += 1
            count[1] += int(rank == 0)


def ranked_graph(graph, votes, k=3):
    # New evaluation export: window proposal count, then p1 count, then label.
    edges = []
    for (s, t), candidates in sorted(votes.items()):
        ranked = sorted(candidates, key=lambda p: (-candidates[p][0], -candidates[p][1], p))
        edges.append({'src': s, 'tgt': t, 'c_edge': ranked[:k],
                      'candidate_support': {p: candidates[p] for p in ranked}})
    return {'nodes': graph['nodes'], 'edges': edges}
