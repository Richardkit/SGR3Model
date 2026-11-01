"""Build a NEW scene-disjoint IVF index from existing per-scan embeddings (CPU)."""
import argparse
import json
import pickle
import os
from pathlib import Path
from common import excluded_scans, scene_map, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'configs/ready_pilot.json'))
    p.add_argument('--nlist', type=int, default=4096)
    p.add_argument('--training-size', type=int, default=300000)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    meta = json.loads(Path(cfg['scene_metadata']).read_text())
    mapping = scene_map(meta)
    excluded = excluded_scans(meta, Path(cfg['evaluation_split']).read_text().split())
    scans = sorted(d for d in Path(cfg['embedding_root']).iterdir()
                   if d.is_dir() and (d / 'frame_patch_embeddings.npy').is_file() and (d / 'patch_keys.pkl').is_file())
    if any(d.name not in mapping for d in scans):
        raise ValueError('Embedding scan not found in scene metadata')
    scans = [d for d in scans if d.name not in excluded]
    if not scans:
        raise ValueError('No eligible KB scans')
    import faiss
    import numpy as np
    faiss.omp_set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    faiss.cvar.distance_compute_blas_threshold = cfg.get('faiss_blas_threshold', 20)
    rng = np.random.default_rng(0)
    print('Eligible scans:', len(scans), 'excluded scene IDs:', len(excluded), flush=True)
    pool = []
    per_scan = max(1, args.training_size // len(scans))
    dim = None
    for d in scans:
        x = np.load(d / 'frame_patch_embeddings.npy', mmap_mode='r', allow_pickle=False)
        if x.ndim != 2 or len(x) == 0 or (dim is not None and x.shape[1] != dim):
            raise ValueError('Invalid embedding shape: ' + str(d))
        dim = x.shape[1]
        pool.append(np.array(x[rng.choice(len(x), min(per_scan, len(x)), replace=False)], dtype=np.float32))
    train = np.concatenate(pool)
    if len(train) < args.nlist * 39:
        raise ValueError('Increase training-size or reduce nlist explicitly; too few training vectors')
    faiss.normalize_L2(train)
    if not np.isfinite(train).all():
        raise ValueError('Nonfinite training embeddings')
    print('Training IVF:', train.shape, 'nlist:', args.nlist, flush=True)
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, args.nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(train)
    print('Training finished; adding vectors', flush=True)
    out = Path(cfg['kb_dir'])
    # Deliberately refuse to overwrite/resume an old or partially built index.
    out.mkdir(parents=True, exist_ok=False)
    keys, inputs = [], []
    for d in scans:
        x = np.load(d / 'frame_patch_embeddings.npy', mmap_mode='r', allow_pickle=False)
        with (d / 'patch_keys.pkl').open('rb') as f:
            local_keys = pickle.load(f)
        if len(local_keys) != len(x) or not all(isinstance(k, str) and k.startswith('frame-') for k in local_keys):
            raise ValueError('Embedding/key mismatch: ' + str(d))
        for start in range(0, len(x), 100000):
            block = np.array(x[start:start+100000], dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite embedding: ' + str(d))
            faiss.normalize_L2(block)
            index.add(block)
        keys.extend(f'{d.name}|{k}' for k in local_keys)
        inputs.append({'scan': d.name, 'rows': len(x),
                       'vectors_sha256': sha256(d / 'frame_patch_embeddings.npy'),
                       'keys_sha256': sha256(d / 'patch_keys.pkl')})
        if len(inputs) % 100 == 0:
            print('Indexed scans:', len(inputs), 'vectors:', index.ntotal, flush=True)
    if index.ntotal != len(keys):
        raise RuntimeError('Index/key count mismatch')
    index.nprobe = cfg['nprobe']
    faiss.write_index(index, str(out / 'index.faiss'))
    with (out / 'keys.pkl').open('xb') as f:
        pickle.dump(keys, f, protocol=4)
    write_json(out / 'manifest.json', {'included_scans': [d.name for d in scans],
        'excluded_scene_scans': sorted(excluded), 'split_sha256': sha256(cfg['evaluation_split']),
        'metadata_sha256': sha256(cfg['scene_metadata']), 'builder_sha256': sha256(__file__),
        'index_type': 'IndexIVFFlat', 'dimension': dim, 'nlist': args.nlist, 'training_rows': len(train),
        'omp_threads': faiss.omp_get_max_threads(),
        'distance_compute_blas_threshold': faiss.cvar.distance_compute_blas_threshold,
        'seed': 0, 'ntotal': index.ntotal, 'source_embeddings': inputs,
        'artifacts': {n: sha256(out / n) for n in ['index.faiss', 'keys.pkl']}})
    print('Built new scene-disjoint KB:', out, 'scans:', len(scans), 'keys:', len(keys))


if __name__ == '__main__':
    main()
