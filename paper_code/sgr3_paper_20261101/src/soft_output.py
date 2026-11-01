"""Recover only complete JSON records, using the legacy extraction helper, without GT."""
import json
from dashscope_client import parse_json
import re

def _extract_array_objects(raw, key):
    m = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not m:
        return []
    i = m.end()
    objs, depth, start = [], 0, None
    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(raw[start:i+1])
                start = None
        elif ch == "]":
            if depth == 0:
                break
        i += 1
    return objs

def parse_model_output(response, main=True):
    text=response['text'];notes=[]
    try:delta=parse_json(text)
    except (json.JSONDecodeError,ValueError):
        required=['new_nodes','new_edges'] if main else ['new_edges']
        if not all('"'+key+'"' in text for key in required):raise ValueError('No delta arrays to recover; refusing global-graph substitution')
        delta={};discarded=0
        for key in required:
            records=[]
            for fragment in _extract_array_objects(text,key):
                try:records.append(json.loads(fragment))
                except (json.JSONDecodeError,ValueError):discarded+=1
            delta[key]=records
        if not any(delta.values()):raise ValueError('No complete model records could be recovered')
        notes.append(dict(action='recover_complete_json_records',discarded_fragments=discarded))
    if not isinstance(delta,dict) or not isinstance(delta.get('new_edges'),list):raise ValueError('Missing model edge list')
    if main and not isinstance(delta.get('new_nodes'),list):raise ValueError('Missing model node list')
    if response['finish_reason']!='stop':notes.append(dict(action='keep_complete_records_from_partial_response',finish_reason=response['finish_reason']))
    return delta,notes
