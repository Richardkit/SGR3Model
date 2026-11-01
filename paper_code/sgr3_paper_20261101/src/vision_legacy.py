# Functions extracted from the working tree; see source_manifest.json.
import os, re, json, glob
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from copy import deepcopy
FRAME_RE_SEQ = re.compile(r"frame-(\d+)\.color\.(?:jpg|jpeg|png)$", re.IGNORECASE)
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from graph_legacy import atomic_write_json
RETRIEVAL_TEMPERATURE = 0.5
UNWEIGHTED = False
GRID_ROWS, GRID_COLS = 4, 6
device = 'cuda'
SIGLIP_REVISION = None

def load_siglip(model_name="google/siglip2-base-patch16-224", device="cuda"):
    processor = SiglipImageProcessor.from_pretrained(model_name, revision=SIGLIP_REVISION)
    model = SiglipVisionModel.from_pretrained(model_name, revision=SIGLIP_REVISION).to(device).eval()
    return processor, model

def parse_kb_key_to_scan_frame(key: str) -> Tuple[str, int]:
    s = key.decode("utf-8", "ignore") if isinstance(key, (bytes, bytearray)) else str(key)
    s = s.replace("\\", "/")

    # 1) UUID
    m_uuid = UUID_RE.search(s)
    if not m_uuid:
        raise ValueError(f"No UUID in key: {s}")
    scan_id = m_uuid.group(0).lower()

    # 2) frame id (fallback for multiple formats)
    m = (
        re.search(r"frame-(\d+)", s, re.IGNORECASE)
        or re.search(r"frame[_=](\d+)", s, re.IGNORECASE)
        or re.search(r"\b(\d{3,6})\b", s)
    )
    if not m:
        raise ValueError(f"No frame id in key: {s}")

    frame_id = int(m.group(1))
    return scan_id, frame_id

def _merge_reference_sgs(ref_jsons: List[Dict[str, Any]],
                         winning_scan: str,
                         keep_policy: str = "most_edges",
                         max_edges_per_object: Optional[int] = None) -> Dict[str, Any]:
    by_id: Dict[int, List[Tuple[Dict[str, Any], int]]] = defaultdict(list)
    frames_used = set()

    for r in ref_jsons:
        sg = r.get("json") or {}
        f_id = int(r.get("kb_frame_id"))
        frames_used.add(f_id)
        for obj in sg.get("objects", []) or []:
            oid = obj.get("id")
            if oid is None:
                continue
            by_id[oid].append((obj, f_id))

    kept_pre: Dict[int, Dict[str, Any]] = {}
    for oid, lst in by_id.items():

        if keep_policy == "most_edges":
            rep = max(lst, key=lambda x: len((x[0].get("edges") or [])))
        elif keep_policy == "min_frame":
            rep = min(lst, key=lambda x: x[1])
        else:
            rep = lst[0]
        kept_pre[oid] = deepcopy(rep[0])

        for i,x in enumerate(lst):
            for edge in x[0]["edges"]:
                if "edges" not in kept_pre[oid]:
                    kept_pre[oid]["edges"] = []
                kept_pre[oid]["edges"].append(edge)


    kept = {oid: o for oid, o in kept_pre.items()}
    kept_ids = set(kept.keys())
    # print(f"[Debug] kept object ids: {kept_ids}")
    # print(f"[Debug] kept objects: {kept}")
    new_edges, seen = [], set()
    for oid, obj in kept.items():
        edges = obj.get("edges", []) or []
        for e in edges:
            src_name = e.get("src_label") if "src_label" in e else ""
            src = e.get("src") if "src" in e else None
            tgt_name = e.get("tgt_label") if "tgt_label" in e else ""
            tgt = e.get("tgt") if "tgt" in e else None
            rel = (e.get("relationship") or e.get("pred_label") or e.get("c_edge") or "").strip()
            if not rel or src is None or tgt is None:
                continue
            if (src in kept_ids) and (tgt in kept_ids):
                key = (src_name, rel, tgt_name)
                if key not in seen:
                    new_edges.append({"src": str(src_name), "tgt": str(tgt_name), "c_edge": rel})
                    seen.add(key)

    if isinstance(max_edges_per_object, int) and max_edges_per_object > 0:
        # simple global cap
        new_edges = new_edges[:max_edges_per_object]

    merged = {
        "scan": winning_scan,
        "frames_used": sorted(frames_used),
        "edges": new_edges
    }
    return merged

@torch.no_grad()
def embed_image_patches_siglip2(model_bundle, img: Image.Image) -> np.ndarray:
    """
    Return patch embeddings: [P, D] float32, L2-normalized.
    P = GRID_ROWS * GRID_COLS (e.g., 24)
    """
    processor, vision_model = model_bundle
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # AMP is optional but recommended (faster)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = vision_model(pixel_values=pixel_values)
    tok = out.last_hidden_state  # [1, T, D]

    # infer H,W from config
    H = W = int(vision_model.config.image_size // vision_model.config.patch_size)
    HW = H * W
    T = tok.shape[1]

    # Some versions of SigLIP2 have no CLS token; determine by length
    if T == HW:
        tokens = tok
    elif T == HW + 1:
        tokens = tok[:, 1:, :]
    else:
        raise RuntimeError(f"Unexpected token length T={T}, expected {HW} or {HW+1}")

    # [1, HW, D] -> [1, D, H, W]
    feat = tokens.reshape(1, H, W, -1).permute(0, 3, 1, 2)

    # pool to GRID
    pooled = F.adaptive_avg_pool2d(feat, output_size=(GRID_ROWS, GRID_COLS))  # [1,D,R,C]
    pooled = pooled.permute(0, 2, 3, 1).reshape(1, GRID_ROWS * GRID_COLS, -1) # [1,P,D]

    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled[0].detach().cpu().numpy().astype(np.float32)  # [P,D]

def run_rag_for_window(
    sequence_dir: str,
    faiss_index,
    rgb_keys,
    kb_scan_root: str,
    query_rgbs: List[str],
    model_bundle,
    out_json_path: str,
    faiss_topk: int = 5,          # final number of reference frames to keep
    patch_topk: int = 10,         # number of candidate patches retrieved from faiss per query patch
    max_frames_per_scan: Optional[int] = None,   # prevent one scan from monopolizing top frames; None means no limit
    *,
    excluded_scan_ids,
    known_scan_ids,
    query_patch_cache=None,
    keys_are_normalized=False,
) -> List[Dict[str, Any]]:

    query_sid = os.path.basename(os.path.dirname(os.path.normpath(sequence_dir)))
    if query_sid not in excluded_scan_ids or query_sid not in known_scan_ids:
        raise ValueError('Retrieval requires verified test-scene exclusion metadata')
    excluded_hits = 0
    if not query_rgbs:
        raise FileNotFoundError(f"No query RGBs provided for {sequence_dir}")

    if not keys_are_normalized:
        rgb_keys = [
            (k.decode("utf-8", "ignore") if isinstance(k, (bytes, bytearray)) else str(k))
            for k in rgb_keys
        ]

    # ----------------------------
    # 1) For each query image: extract patch embeddings [P,D]
    # ----------------------------
    query_patch_list = []
    for p in query_rgbs:
        if query_patch_cache is not None:
            if p not in query_patch_cache:
                raise ValueError('Query image absent from the verified patch cache')
            Q = np.asarray(query_patch_cache[p], dtype=np.float32)
            if not np.isfinite(Q).all() or not np.allclose(np.linalg.norm(Q, axis=1), 1, atol=1e-3):
                raise ValueError('Cached query patches must be finite and normalized')
        else:
            img = Image.open(p).convert("RGB")
            Q = embed_image_patches_siglip2(model_bundle, img)  # [P,D], normalized
        if Q.ndim != 2:
            raise RuntimeError(f"Bad query patch emb shape: {Q.shape} for {p}")
        query_patch_list.append(Q)

    # Global accumulated frame score: (scan_id, frame_id) -> score
    frame_score_all = defaultdict(float)

    # Optional: keep per-query-image per-frame scores for analysis
    qimg_scores = []

    # ----------------------------
    # 2) For each query image: FAISS batch retrieval (by patch)
    #    compute weighted MaxSim for each kb frame
    # ----------------------------
    for qi, Q in enumerate(query_patch_list):
        sims, idxs = faiss_index.search(Q.astype(np.float32), patch_topk)

        P, D = Q.shape

        # ---------- Preserve original patch weights ----------
        temperature = RETRIEVAL_TEMPERATURE
        S = Q @ Q.T
        mask = np.ones((P, P), dtype=bool)
        np.fill_diagonal(mask, False)
        cor_mean = S[mask].reshape(P, P - 1).mean(axis=1)
        score = -cor_mean / temperature
        score = score - score.max()
        weights = np.exp(score)
        weights = weights / (weights.sum() + 1e-12)
        if UNWEIGHTED:
            weights = np.ones(P, dtype=np.float32) / P

        # per-kb-frame: record the best sim for each query patch on that frame
        per_img_max = {}

        for p_idx in range(P):
            for r in range(patch_topk):
                kb_index = int(idxs[p_idx, r])
                if kb_index < 0:
                    continue

                sim = float(sims[p_idx, r])
                key = rgb_keys[kb_index]

                try:
                    kb_sid, kb_fid = parse_kb_key_to_scan_frame(key)
                except Exception as exc:
                    raise ValueError('Invalid KB key in search results') from exc
                if kb_sid not in known_scan_ids:
                    raise ValueError('Retrieved scan is absent from scene metadata')
                if kb_sid in excluded_scan_ids:
                    excluded_hits += 1
                    continue

                frame_key = (kb_sid, int(kb_fid))

                arr = per_img_max.get(frame_key)
                if arr is None:
                    arr = np.full((P,), -1e9, dtype=np.float32)
                    per_img_max[frame_key] = arr

                if sim > arr[p_idx]:
                    arr[p_idx] = sim

        # Compute weighted MaxSim score of this query image against all kb frames
        per_frame_score = {}
        for frame_key, arr in per_img_max.items():
            valid = arr > -1e8
            sc = float((arr[valid] * weights[valid]).sum())
            per_frame_score[frame_key] = sc
            frame_score_all[frame_key] += sc

        qimg_scores.append(per_frame_score)

    if not frame_score_all:
        raise RuntimeError("No eligible reference frames after scene exclusion; refusing fallback.")

    # ----------------------------
    # 3) Directly select global top reference frames
    #    optional: limit max frames per scan to max_frames_per_scan
    # ----------------------------
    ranked_frames = sorted(
        frame_score_all.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_frames = []
    per_scan_counter = defaultdict(int)

    for (sid, fid), sc in ranked_frames:
        if max_frames_per_scan is not None and per_scan_counter[sid] >= max_frames_per_scan:
            continue
        top_frames.append((sid, fid, sc))
        per_scan_counter[sid] += 1
        if len(top_frames) >= faiss_topk:
            break

    if not top_frames:
        raise RuntimeError("Top frame selection failed; no valid reference frames found.")

    print("[RAG-FrameVote] top reference frames:")
    for sid, fid, sc in top_frames:
        print(f"  scan={sid}, frame={fid}, score={sc:.4f}")

    # ----------------------------
    # 4) Load SGs of top reference frames and merge edges
    # ----------------------------
    chosen_refs = []
    for sid, fid, sc in top_frames:
        if sid in excluded_scan_ids:
            raise RuntimeError('Excluded scene reached reference loading')
        json_path = os.path.join(kb_scan_root, sid, "frame_sg", f"frame-{fid:06d}.json")
        if not os.path.isfile(json_path):
            print(f"[INFO] missing frame json -> skip: {json_path}")
            continue
        try:
            with open(json_path, "r") as f:
                sg = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load json ({json_path}): {e}")
            continue

        chosen_refs.append({
            "kb_scan_id": sid,
            "kb_frame_id": int(fid),
            "json_path": json_path,
            "json": sg,
            "distance": float(sc),   # keep field name for compatibility for now
        })

    if not chosen_refs:
        raise RuntimeError("No valid reference frame SG json could be loaded.")

    # Provide a dummy winning_scan for compatibility with the original merge function
    # Numeric object IDs are local to a scan. Merge within each scan first,
    # then combine semantic triples, never merge objects across scan IDs.
    merged_edges = merge_reference_scenes(chosen_refs)

    # ----------------------------
    # 5) Save analysis results
    # ----------------------------
    out = {
        "query_scan": query_sid,
        "query_patch_source": "precomputed" if query_patch_cache is not None else "gpu_encoder",
        "patch_topk": patch_topk,
        "reference_frame_topk": faiss_topk,
        "selection_policy": "global_frame_sum_topk_v1",
        "retrieval_temperature": RETRIEVAL_TEMPERATURE,
        "excluded_scan_ids": sorted(excluded_scan_ids),
        "excluded_patch_hits": excluded_hits,
        "query_images": list(query_rgbs),
        "loaded_reference_frames": [
            {"scan_id": r['kb_scan_id'], "frame_id": r['kb_frame_id'], "json_path": r['json_path']}
            for r in chosen_refs
        ],
        "top_reference_frames": [
            {"scan_id": sid, "frame_id": int(fid), "score": float(sc)}
            for sid, fid, sc in top_frames
        ],
        "reference_edges": merged_edges,
    }
    atomic_write_json(out_json_path, out)
    print(f"[RAG-FrameVote] saved -> {out_json_path}")

    return merged_edges


def merge_reference_scenes(chosen_refs):
    by_scan = defaultdict(list)
    for ref in chosen_refs:
        by_scan[ref['kb_scan_id']].append(ref)
    edges, seen = [], set()
    for sid, refs in by_scan.items():
        graph = _merge_reference_sgs(refs, winning_scan=sid)
        for edge in graph['edges']:
            key = (edge['src'], edge['tgt'], edge['c_edge'])
            if key not in seen:
                seen.add(key)
                edges.append(edge)
    return edges
