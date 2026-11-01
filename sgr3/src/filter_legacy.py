# Functions extracted from the working tree; see source_manifest.json.
import os, re, json, glob
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from copy import deepcopy
FRAME_RE_SEQ = re.compile(r"frame-(\d+)\.color\.(?:jpg|jpeg|png)$", re.IGNORECASE)
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from huggingface_hub import hf_hub_download
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
DATA_ROOT = None
COLQWEN_REVISION = None
COLQWEN_BASE_NAME = 'vidore/colqwen2.5-base'
COLQWEN_BASE_REVISION = '92908120384b7a2110c5beda3ab29cbdb2c08e49'

def find_indices_by_substrings(tokens, frames):
    """
    get indices of tokens whose paths contain any of the given frames
    """
    out = []
    for f in frames:
        hit = [i for i, p in enumerate(tokens) if f in p]
        if not hit:
            print(f"[WARN] cannot find any path contains '{f}'")
            out.append(None)
        else:
            out.append(hit[0])
    return out

def l2norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n

def cheap_image_proxy(tokens_3d, mode="mean"):
    """
    tokens_3d: (N,T,D) fp16/32
    return: (N,D) float32 normalized
    """
    X = tokens_3d.astype(np.float32, copy=False)
    if mode == "mean":
        v = X.mean(axis=1)           # (N,D)
    elif mode == "max":
        v = X.max(axis=1)
    else:
        raise ValueError(f"unknown CHEAP_PROXY={mode}")
    v = l2norm_rows(v)
    return v

def topM_by_cosine(proxy, M = 30):
    """
    proxy: (N,D) row-normalized
    returns indices: (N,M), sims: (N,M)
    """
    S = proxy @ proxy.T
    np.fill_diagonal(S, -1.0)
    M = min(M, S.shape[0]-1)
    if M <= 0:
        return np.empty((S.shape[0], 0), dtype=int), np.empty((S.shape[0], 0))
    idx = np.argpartition(-S, kth=M-1, axis=1)[:, :M]
    sims = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-sims, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    sims = np.take_along_axis(sims, order, axis=1)

    return idx, sims

def compute_sim_for_window(scan_id, frames: List[str], image_buffer: List[str], CHEAP_PROXY="mean", PRE_SIM_THRESH=0.45, topk= 5):

    # topk is the number of most similar images to return for each frame in frames
    sig_dir = os.path.join(DATA_ROOT, scan_id, "siglip2_val")
    tokens_path = os.path.join(sig_dir, "patch_tokens_fp16.npy")
    paths_file_path = os.path.join(sig_dir, "paths.json")

    # open tokens file
    tokens = np.load(tokens_path, mmap_mode="r")  # (N,T,D)
    paths = json.load(open(paths_file_path, "r"))  # list of N paths
    N, T, D = tokens.shape
    paths = [os.path.join(DATA_ROOT, scan_id, "sequence", os.path.basename(p)) for p in paths]
    if len(paths) != N:
        raise ValueError("SigLIP cache/path count mismatch")

    tgt_idxs = find_indices_by_substrings(paths, frames)
    image_buffer = [os.path.basename(p) for p in image_buffer] # save absolute paths of finished query images
    buf_idxs = find_indices_by_substrings(paths, image_buffer)

    # pre-load tokens for targets into RAM (small)
    V = {}
    for i in tgt_idxs:
        if i is None:
            continue
        Vi = np.array(tokens[i], dtype=np.float32)  # (T,D)
        Vi = Vi / (np.linalg.norm(Vi, axis=1, keepdims=True) + 1e-8)  # L2-normalize
        V[i] = Vi

    proxy = cheap_image_proxy(tokens, mode=CHEAP_PROXY)  # (N,D)
    cand_idx, cand_sim = topM_by_cosine(proxy)
    out = {}
    
    for i in tgt_idxs:

        if i is None:
            continue

        Vi = np.array(tokens[i], dtype=np.float32)
        Vi = Vi / (np.linalg.norm(Vi, axis=1, keepdims=True) + 1e-8)
        scores = []

        for j, sim in zip(cand_idx[i],cand_sim[i]):

            if j in buf_idxs:

                if sim < PRE_SIM_THRESH:
                    continue

                scores.append((int(j), float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)

        topk_id = scores[:min(topk, len(scores))] 
        out[int(i)] = [paths[k[0]] for k in topk_id]

    return out, tgt_idxs

@torch.no_grad()
def colqwen_score_image_to_images(processor, model, query_img_path: str, cand_img_paths: list[str]) -> list[float]:
    # query as 1-image batch
    q_img = Image.open(query_img_path).convert("RGB")
    c_imgs = [Image.open(p).convert("RGB") for p in cand_img_paths]

    batch_q = processor.process_images([q_img]).to(model.device)
    batch_c = processor.process_images(c_imgs).to(model.device)

    q_emb = model(**batch_q)      # multi-vector embeddings
    c_emb = model(**batch_c)

    raw = processor.score_multi_vector(q_emb, c_emb)[0]  # (Nc,) sum over Tq
    Tq = q_emb.shape[1]
    scores = (raw / Tq).detach().cpu().tolist()
    return scores

def load_colqwen(COLQWEN_NAME: str, device="cuda"):
    # ColQwen is a LoRA adapter. Pin adapter and base separately: passing the
    # adapter commit as the base revision breaks Transformers' automatic loader.
    adapter_config = json.load(open(hf_hub_download(COLQWEN_NAME, 'adapter_config.json',
                                                  revision=COLQWEN_REVISION)))
    if adapter_config['base_model_name_or_path'] != COLQWEN_BASE_NAME:
        raise ValueError('Unexpected ColQwen base model; review its revision first')
    model = ColQwen2_5.from_pretrained(
        COLQWEN_BASE_NAME,
        revision=COLQWEN_BASE_REVISION,
        torch_dtype=torch.bfloat16,   # or float16 if needed
        device_map="cuda:0" if device.startswith("cuda") else "cpu",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
    mapping = model._checkpoint_conversion_mapping
    model.load_adapter(COLQWEN_NAME, revision=COLQWEN_REVISION,
                       adapter_kwargs={'key_mapping': mapping})
    # Transformers only warns on missing LoRA keys. Reject partial loads and
    # compare every tensor with the pinned checkpoint before using any scores.
    checkpoint = load_file(hf_hub_download(COLQWEN_NAME, 'adapter_model.safetensors',
                                          revision=COLQWEN_REVISION))
    expected = {}
    for key, value in checkpoint.items():
        key = key.removeprefix('base_model.model.')
        for pattern, replacement in mapping.items():
            key, replacements = re.subn(pattern, replacement, key)
            if replacements:
                break
        if key in expected:
            raise RuntimeError('Adapter key conversion collision')
        expected[key] = value
    actual = get_peft_model_state_dict(model, adapter_name='default')
    if set(actual) != set(expected):
        raise RuntimeError(f'ColQwen adapter tensor names differ: missing={len(set(expected)-set(actual))}, '
                           f'unexpected={len(set(actual)-set(expected))}')
    for key, value in expected.items():
        loaded = actual[key].detach().cpu()
        if not torch.equal(loaded, value.to(loaded.dtype)):
            raise RuntimeError('ColQwen adapter tensor did not load correctly: ' + key)
    model._robio_adapter_validation = {'tensor_count': len(expected),
                                      'all_exact_after_dtype_conversion': True}
    model.eval()
    processor = ColQwen2_5_Processor.from_pretrained(COLQWEN_NAME, revision=COLQWEN_REVISION)
    return processor, model

def img_if_new_check_colqwen(
    window_img_paths,
    scan_id,
    image_buffer,
    colqwen_model,
    colqwen_proc,
    device="cuda",
    COLQWEN_SIM_THRESH=0.5,
    topk_buf=5,
):
    # Step 1: SigLIP prefilter (reuse your existing code)
    frames = [os.path.basename(p) for p in window_img_paths]
    rag_sim_imgs_window, tgt_idxs = compute_sim_for_window(
        scan_id, frames, image_buffer
    )

    keep_imgs = []

    for img_path, global_i in zip(window_img_paths, tgt_idxs):

        # Fail-open
        if global_i is None:
            keep_imgs.append(img_path)
            continue

        buf_imgs = rag_sim_imgs_window.get(global_i, [])[:topk_buf]

        # No buffer → new
        if len(buf_imgs) == 0:
            keep_imgs.append(img_path)
            continue

        # Step 2: ColQwen similarity
        scores = colqwen_score_image_to_images(colqwen_proc, colqwen_model, img_path, buf_imgs)
        max_sim = max(scores) if scores else -1e9
        print(f"[NEW_CHECK] {os.path.basename(img_path)} max_sim={max_sim:.4f} keep={max_sim < COLQWEN_SIM_THRESH}")
        if max_sim < COLQWEN_SIM_THRESH:
            keep_imgs.append(img_path)
        # else: repeat → drop

    return keep_imgs
