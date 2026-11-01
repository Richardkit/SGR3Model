"""Sealed, per-policy retrieval artifacts. Consumer has no numpy/FAISS/model imports."""
import hashlib
import json
from pathlib import Path
from common import sha256, write_json


def signature(cfg):
    if cfg.get("retrieval_selection_policy") != "global_frame_sum_topk_v1":
        raise ValueError("Paper package requires cumulative global frame ranking, not scene argmax")
    relevant = {k: v for k, v in cfg.items() if k != 'retrieval_cache_dir'}
    # Only the Qwen image-input ablation may share original-window retrieval.
    # Every retrieval setting and input checksum remains checked unchanged.
    if relevant.get('generation_image_policy') == 'filtered':
        relevant['generation_image_policy'] = 'original_window'
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seal(row):
    row = dict(row)
    row.pop('record_sha256', None)
    row['record_sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return row


def read_sealed(path):
    row = json.loads(Path(path).read_text())
    if row.get('record_sha256') != seal(row)['record_sha256']:
        raise ValueError('Retrieval cache checksum mismatch: ' + str(path))
    return row


class CachedRetrieval:
    def __init__(self, cfg, sid, frames):
        self.root = Path(cfg['retrieval_cache_dir'])
        self.manifest = read_sealed(self.root / 'manifest.json')
        m = self.manifest
        if m['status'] != 'complete' or m['signature'] != signature(cfg):
            raise ValueError('Retrieval cache incomplete or configuration differs')
        if set(m['scans']) != set(Path(cfg['evaluation_split']).read_text().split()):
            raise ValueError('Retrieval cache does not cover all test scans')
        for key, digest in m['input_sha256'].items():
            if sha256(cfg[key]) != digest:
                raise ValueError('Retrieval cache input changed: ' + key)
        scanpath = self.root / sid / 'manifest.json'
        if sha256(scanpath) != m['scans'][sid]:
            raise ValueError('Retrieval scan manifest changed')
        self.scan = read_sealed(scanpath)
        if self.scan['frames'] != frames:
            raise ValueError('Retrieval cache frame order mismatch')
        for image, digest in self.scan['image_sha256'].items():
            if sha256(image) != digest:
                raise ValueError('Image changed after retrieval')
        self.sid = sid
        self.provenance = {'retrieval_cache': str(self.root), 'manifest_sha256': sha256(self.root/'manifest.json'),
                           'scan_manifest_sha256': sha256(scanpath), 'gpu_required': False,
                           'index_loaded': False}

    def row(self, wi, original):
        entry = self.scan['windows'][wi]
        path = self.root / self.sid / entry['file']
        if sha256(path) != entry['sha256']:
            raise ValueError('Retrieval window changed')
        row = read_sealed(path)
        if row['window'] != wi or row['scan'] != self.sid or row['signature'] != self.manifest['signature'] or row['original_images'] != original:
            raise ValueError('Retrieval window identity mismatch')
        return row

    def kept_images(self, wi, original):
        return self.row(wi, original)['kept_images']

    def retrieve(self, wi, original, kept, output, blocked):
        row = self.row(wi, original)
        r = row['retrieval']
        if r is not None and r.get('selection_policy') != 'global_frame_sum_topk_v1':
            raise ValueError('Wrong retrieval algorithm in cached window')
        if row['kept_images'] != kept or r is None or r['query_images'] != kept or set(r['excluded_scan_ids']) != blocked:
            raise ValueError('Retrieval policy/window mismatch')
        if any(ref['scan_id'] in blocked for ref in r['top_reference_frames'] + r['loaded_reference_frames']):
            raise ValueError('Forbidden reference in cache')
        for path, digest in row['reference_sources_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Reference graph changed since retrieval')
        write_json(output, r)
        write_json(str(output)+'.cache.json', {'cache':str(self.root/self.sid/f'win{wi:04d}.json'),
                                            'record_sha256':row['record_sha256'], 'index_loaded':False})
        return r['reference_edges']
