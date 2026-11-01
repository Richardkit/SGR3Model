# Functions extracted from the working tree; see source_manifest.json.
import os, re, json, glob
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from copy import deepcopy
FRAME_RE_SEQ = re.compile(r"frame-(\d+)\.color\.(?:jpg|jpeg|png)$", re.IGNORECASE)
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

def atomic_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def list_sequence_rgbs(sequence_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(sequence_dir, "frame-*.color.jpg")) \
          + glob.glob(os.path.join(sequence_dir, "frame-*.color.png")) \
          + glob.glob(os.path.join(sequence_dir, "frame-*.color.jpeg"))
    def _idx(p):
        m = FRAME_RE_SEQ.search(os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=_idx)

def plan_windows(n: int, win: int, stride: int) -> List[Tuple[int, int]]:
    out = []
    i = 0
    while i < n:
        j = min(i + win, n)
        out.append((i, j))
        if j == n:
            break
        i += stride
    return out

def load_vocab_list(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(s)
    return items

def normalize_text(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", (s or "").strip().lower())

def _to_int_safe(x):
    try:
        return int(str(x))
    except Exception:
        return None

def build_id_map(local_nodes, global_nodes_by_id):
    """
    New build_id_map:
    - No longer do similarity matching based on name/description
    - Fully trust that the model-generated node["id"] is already the 'global id it considers'
    - If the id already exists in global_nodes_by_id, treat it as the same object and reuse directly
    - If the id does not exist:
        * If the id is an integer greater than the current maximum used id, use it directly as the new global id
        * Otherwise, to avoid conflicts, assign a new incrementing id (next_id) and map the local id to it
    """
    id_map = {}
    new_global_nodes = []

    # Count currently occupied integer ids in the global graph
    existing_int_ids = []
    for gid in global_nodes_by_id.keys():
        v = _to_int_safe(gid)
        if v is not None:
            existing_int_ids.append(v)
    max_existing = max(existing_int_ids) if existing_int_ids else 0
    next_id = max_existing + 1

    used_ids = set(global_nodes_by_id.keys())  # in string form

    for node in local_nodes:
        lid_raw = node.get("id")
        lid = str(lid_raw).strip()
        name = node.get("name", "")
        desc = node.get("description", "")

        # Case 1: the model-provided id already exists in the global graph → treat as the same object directly
        if lid in global_nodes_by_id:
            id_map[lid] = lid
            # If you want to update the global description, you can also merge description here
            # e.g., append the new desc (optional)
            continue

        # Case 2: this id has not been used yet
        lid_int = _to_int_safe(lid)
        if lid_int is not None and lid_int > max_existing and lid not in used_ids:
            # The model provided an integer larger than all current ids; treat as a valid new global id
            g_id = lid
            used_ids.add(g_id)
            # Also update max_existing / next_id to avoid future conflicts
            if lid_int > max_existing:
                max_existing = lid_int
                next_id = max_existing + 1
        else:
            # The model provided a small already-used integer / non-integer id / other unusual form;
            # for safety, we assign a truly unique new_id
            g_id = str(next_id)
            next_id += 1
            used_ids.add(g_id)

        id_map[lid] = g_id
        new_global_nodes.append({
            "id": g_id,
            "name": name,
            "description": desc,
        })

    return id_map, new_global_nodes

def merge_graphs(global_graph: Dict[str, Any], local_graph: Dict[str, Any]) -> Dict[str, Any]:

    if "nodes" not in global_graph:
        global_graph["nodes"] = []
    if "edges" not in global_graph:
        global_graph["edges"] = []

    global_nodes_by_id = {str(n["id"]): n for n in global_graph["nodes"]}

    local_nodes = (local_graph.get("nodes") or [])
    local_edges = (local_graph.get("edges") or [])

    # --- normalize local (no seen_in_frames) ---
    for n in local_nodes:
        n["id"] = str(n.get("id"))
        # Keep only required fields
        n["name"] = n.get("name", "")
        n["description"] = n.get("description", "")

    for e in local_edges:
        # Normalize the relationship field
        if "c_edge" not in e:
            e["c_edge"] = e.get("predicate") or e.get("relation") or ""

    # Build local->global mapping + new nodes
    id_map, new_global_nodes = build_id_map(local_nodes, global_nodes_by_id)

    # ---- Merge nodes (without seen_in_frames) ----
    for ln in local_nodes:
        ln_id  = str(ln.get("id"))
        ln_name = ln.get("name", "")
        ln_desc = ln.get("description", "")
        gid = id_map[ln_id]

        if gid in global_nodes_by_id:
            cur = global_nodes_by_id[gid]
            if not isinstance(cur, dict):
                # Fallback: if the historical value is abnormal/None, overwrite with the full node
                cur = {"id": gid, "name": ln_name, "description": ln_desc}
                # global_nodes_by_id[gid] = cu

            # Fill in name (force rename)
            if ln_name:
                cur["name"] = ln_name
            # only update when the description is more “informative”
            if len(normalize_text(ln_desc)) > len(normalize_text(cur.get("description", ""))):
                cur["description"] = ln_desc
        else:
            # New global node: only three fields
            global_nodes_by_id[gid] = {"id": gid, "name": ln_name, "description": ln_desc}

    # ---- Merge edges and deduplicate ----
    # prepare existing edges: map (src,tgt) -> edge index
    pair2idx = {}
    for idx, e in enumerate(global_graph["edges"]):
        src = str(e.get("src")).strip()
        tgt = str(e.get("tgt")).strip()
        pair2idx[(src, tgt)] = idx

    # merge
    for e in local_edges:
        src_l = str(e.get("src", "")).strip()
        tgt_l = str(e.get("tgt", "")).strip()

        src_g = id_map.get(src_l)
        tgt_g = id_map.get(tgt_l)
        if not src_g or not tgt_g:
            # edge refers to an id not present in local nodes (hallucinated / missing stub)
            continue

        if src_g == "" or tgt_g == "":
            continue
        
        preds = e["c_edge"]

        if isinstance(preds, str):
            preds = [preds]
        if not isinstance(preds, list):
            preds = []


        key = (src_g, tgt_g)
        if key in pair2idx:
            i = pair2idx[key]
            old_preds = global_graph["edges"][i]["c_edge"]
            if isinstance(old_preds, str):
                old_preds = [old_preds]
            merged = list(dict.fromkeys(old_preds + preds))  # union (preserve order), dict keys must be unique
            global_graph["edges"][i]["c_edge"] = merged
        else:
            global_graph["edges"].append({
                "src": src_g,
                "tgt": tgt_g,
                "c_edge": preds
            })
            pair2idx[key] = len(global_graph["edges"]) - 1


    # Sort by numeric ID (global IDs are ensured to be numeric)
    def _sort_key(n):
        s = str(n.get("id", ""))
        return (0, int(s)) if s.isdigit() else (1, s)

    global_graph["nodes"] = sorted(global_nodes_by_id.values(), key=_sort_key)
    return global_graph

def build_prompt_window_rag(
    window_rgbs: List[str],
    prev_global: Optional[Dict[str, Any]],
    reference_edges: List[Dict[str, Any]],
    classes_list: List[str],
    relations_list: List[str],
    predicate_encoding: str = 'labels',
) -> List[Dict[str, Any]]:
    
    import json as _json

    content: List[Dict[str, Any]] = []

    # --- visual inputs ---
    for p in window_rgbs:
        content.append({"type": "image", "image": p})

    # --- stringify context blocks (keep exactly your variable names) ---
    prev_str = _json.dumps(prev_global or {"nodes": [], "edges": []},
                           ensure_ascii=False, separators=(",", ":"))
    refs_str = _json.dumps({"edges": reference_edges or []},
                           ensure_ascii=False, separators=(",", ":"))


    # classes_txt kept for compatibility (even if not used in guide)
    classes_txt   = ", ".join(classes_list)
    relations_txt = ", ".join(relations_list)

    # --------- main guide (uses tags, no huge inline text) ----------
    guide = f"""
You are a reliable 3D scene graph annotator for indoor scenes.

You are given:
- multiple images of ONE scene window, which contains several images,
- a PREVIOUS global scene graph, it contains already detected nodes and edges of the scene same as images of this window (provided in <PREV_GRAPH>),
- reference relations from similar scenes (provided in <REF_EDGES>),
- the allowed predicate list (provided in <ALLOWED_PREDICATES>).
- the allowed object class list (see <ALLOWED_CLASSES>).

You MUST internally follow THREE sequential reasoning steps:
STEP 0: Consistency matching
STEP 1: Node identification
STEP 2: Relation prediction
You MUST output ONLY the final JSON object, without a code fence or end marker.

STEP 0 — CONSISTENCY MATCHING

In this step, you must decide ID consistency BEFORE any node creation.

For each detected object O in this window, even if it is partially visible, you must:

  - Determine O's object class. The class MUST belong to <ALLOWED_CLASSES>

  - Treat clustered small objects of same type (e.g., multiple shoes, books, bottles) as one single node, but treat large objects (e.g., bed, sofa, table, chairs or larger furniture) separately.

  - Decide whether O corresponds to an OLD node from <PREV_GRAPH> or is NEW:

    You must reuse an old id if there is evidence that O is the SAME physical object of a node in <PREV_GRAPH>.

    Evidence types you MUST consider:
      1) similar relative position (left/right/front/behind/closeby to prominent objects), based on current images and edges in <PREV_GRAPH>,
      2) similar visual cues (color, material, shape, text/logo, subparts) based on current images with candidates' descriptions in <PREV_GRAPH>,
      3) if O is a floor, ceiling or door, they are unique per scene, if there is already one in <PREV_GRAPH>, reuse it, otherwise create a new one,
 
  - Record the decision internally:
      1) All objects in this window that match an existing node id and are visible MUST be included in seen_old_nodes.

STEP 1 — NODE IDENTIFICATION

In this step, you must do:

  - Follow the consistency decisions made in STEP 0.
  - For each object marked as OLD in STEP 0:
      * Reuse its existing id from <PREV_GRAPH>.
      * Add ONLY its id to "seen_old_nodes".
      * DO NOT output it as a node.

  - For each object marked as NEW in STEP 0:
      * Create a new node entry in "new_nodes".
      * Assign a new global id strictly larger than MAX_EXISTING_ID in <PREV_GRAPH>.
      * Assign new ids in increasing order.

Rules:
  - "new_nodes" MUST contain ONLY NEW nodes.
  - "seen_old_nodes" MUST contain ONLY ids of reused old nodes.
  - Structural elements (floor, wall, ceiling, window, door, curtain) must be
    included if they appear in the window and were marked as NEW or OLD in STEP 0.
  - Component parts (e.g., chair leg, table leg, cabinet door, bicycle wheel)
    must NOT be separate nodes.

Node fields for NEW nodes:
  - "id": global integer ID
  - "name": short noun phrase (must match <ALLOWED_CLASSES>)
  - "description": visual cues (color, material, shape, parts, text/logo, spatial location, relation to surrounding objects) that help distinguish it from other same-class objects.


STEP 2 — RELATION PREDICTION

In this step, you must do:
    - First, carefully read the <REF_EDGES>, it includes all edges from a visually similar scan, you need to consider which src-tgt pairs have potential edges and what predicates they have.
    - Second, remember all existing edges from <PREV_GRAPH> and newly detected nodes from Step 1.
    - Third, based on the images and the nodes generated from Step 1, decide which (src, tgt) pairs have visually or logically supported relations, considering the experience you learned from <REF_EDGES>. However, DO NOT copy edges blindly from <REF_EDGES>.
    - Fourth, choose possible predicates for each (src, tgt) pair, strictly from the <ALLOWED_PREDICATES>, also consider edges between nodes from <PREV_GRAPH> and newly detected nodes.
    - Generate all possible edges, there are also potential edges between nodes distant from each other.
Rules:
  - Use ONLY predicates from <ALLOWED_PREDICATES>. COPY each label character-for-character.
    Never use synonyms, paraphrases, or grammatical inflections: for example, use "cover", not "covering".
    Likewise use "front", not "in front of". Object class names must also be copied exactly.
  - NEW edges can connect new nodes and old nodes in ANY combination:
      * old -> old (only if newly observed/confirmed in this window)
      * old -> new
      * new -> old
      * new -> new
  - If you output an edge (src,tgt), you MUST output EXACTLY 3 DISTINCT predicates:
      "c_edge": ["p1","p2","p3"]
    If only 1 predicate is obvious, choose 2 additional compatible predicates from the <ALLOWED_PREDICATES>, make sure "p1" is the most confident one.
    (e.g., relative position left/right/front/behind/higher/lower when consistent).
  - "close by" alone is NOT enough; add extra spatial predicates (left/right/front/behind) when possible.
  - Each node must have at least three edges with its cloest three other nodes, for this reason, the edges number should be at least 3 times the nodes number.
  - For nodes on floor, there is must be at "standing on" or "supported by" edge to the floor node
  - For floor and wall, wall nodes must have at least one "attached to" edge to the floor node.

FINAL OUTPUT FORMAT 

Output ONLY one JSON object with exactly these keys:

{{
  "seen_old_nodes":[<INT>,<INT>,...],
  "new_nodes": [
    {{"id": <INT>, "name": "<short noun phrase>", "description": "<text>"}},
    ...
  ],
  "new_edges": [
    {{"src": <INT>, "tgt": <INT>, "c_edge": ["p1","p2","p3"]}},
    ...
  ]
}}

Before returning, check that every class and predicate is an EXACT member of its allowed list,
every c_edge has exactly three DIFFERENT labels, and all edge endpoints are existing or new IDs.
Return ONLY the JSON object. Do not append any end marker.

""".strip()

    if predicate_encoding == 'ids':
        guide = guide.replace('"c_edge": ["p1","p2","p3"]', '"c_edge": [<INT>,<INT>,<INT>]')
        guide = guide.replace('every c_edge has exactly three DIFFERENT labels',
                              'every c_edge has exactly three DIFFERENT integer predicate codes')
        guide += '''\n\nPREDICATE OUTPUT ENCODING (mandatory):
The allowed-predicate table below assigns each predicate a fixed INTEGER code.
For c_edge, output exactly three DISTINCT INTEGER codes from that table, in order of confidence.
Do NOT output predicate words in c_edge. The earlier p1/p2/p3 notation means the three integer codes.
Object node IDs and predicate codes are different: src/tgt identify objects, c_edge identifies predicates.
Example of the required edge syntax: {"src": 1, "tgt": 2, "c_edge": [1, 2, 3]}.
Choose the actual codes using the images and the table; the example is syntax only.
Keep node names as exact object-class labels. Return JSON only.'''
        relations_txt = '\n'.join(f'{i}: {label}' for i, label in enumerate(relations_list, 1))
    elif predicate_encoding != 'labels':
        raise ValueError('Unknown predicate encoding')

# --- append guide first (stable prefix) ---
    content.append({"type": "text", "text": guide})

    # --- append tagged blocks (variable content) ---
    content.append({"type": "text", "text": f"<PREV_GRAPH>\n{prev_str}\n</PREV_GRAPH>"})
    content.append({"type": "text", "text": f"<REF_EDGES>\n{refs_str}\n</REF_EDGES>"})
    content.append({"type": "text", "text": f"<ALLOWED_PREDICATES>\n{relations_txt}\n</ALLOWED_PREDICATES>"})
    content.append({"type": "text", "text": f"<ALLOWED_CLASSES>\n{classes_txt}\n</ALLOWED_CLASSES>"})
    # (Optional) keep classes list if you want it visible to model later; currently not required for edges

    messages = [{"role": "user", "content": content}]
    return messages
