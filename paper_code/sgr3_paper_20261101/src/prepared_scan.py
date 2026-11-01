"""Read verified preprocessing artifacts without importing a GPU model."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from common import sha256

SIGNATURE_KEYS=['window_size','stride','filter_warmup_windows','colqwen_threshold',
                'colqwen_model','colqwen_revision','siglip_model','siglip_revision']


def function_hash(path,names):
    source=Path(path).read_text();tree=ast.parse(source)
    text='\n'.join(ast.get_source_segment(source,n) for n in tree.body
                   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names)
    if not text:raise ValueError('Missing encoder functions')
    return hashlib.sha256(text.encode()).hexdigest()


class PreparedScan:
    def __init__(self,cfg,sid,frames):
        root=Path(cfg['precomputed_cache_dir']);manifest=json.loads((root/'manifest.json').read_text())
        if manifest['status']!='complete':raise ValueError('All test scans must finish preprocessing before generation')
        if manifest['split_sha256']!=sha256(cfg['evaluation_split']):raise ValueError('Precomputed split mismatch')
        ids=Path(cfg['evaluation_split']).read_text().split()
        if set(manifest['scans'])!=set(ids) or len(manifest['completed_scans'])!=len(ids):
            raise ValueError('Preprocessing does not cover the frozen test split')
        signature={k:cfg[k] for k in SIGNATURE_KEYS}
        if manifest['config_signature']!=signature:raise ValueError('Precomputed filter/model settings mismatch')
        src=Path(__file__).parent
        if sha256(src/'filter_legacy.py')!=manifest['source_sha256']['filter_legacy.py']:
            raise ValueError('Filtering source changed since preprocessing')
        original=root/'source_snapshot/vision_legacy.py'
        if sha256(original)!=manifest['source_sha256']['vision_legacy.py']:
            raise ValueError('Encoder snapshot mismatch')
        names={'load_siglip','embed_image_patches_siglip2'}
        if function_hash(original,names)!=function_hash(src/'vision_legacy.py',names):
            raise ValueError('Query encoder changed since preprocessing')
        folder=root/sid;path=folder/'manifest.json'
        if sha256(path)!=manifest['scan_manifests_sha256'][sid]:raise ValueError('Scan manifest mismatch')
        info=json.loads(path.read_text())
        if info['status']!='complete' or info['config_signature']!=signature:raise ValueError('Incomplete scan cache')
        for name,digest in info['artifacts'].items():
            if sha256(folder/name)!=digest:raise ValueError('Corrupted preprocessing artifact: '+name)
        for name,digest in info['proxy_inputs_sha256'].items():
            if sha256(name)!=digest:raise ValueError('Proxy inputs changed after preprocessing')
        names=json.loads((folder/'frames.json').read_text())
        if names!=[Path(f).name for f in frames]:raise ValueError('Cached frame order differs')
        for f in frames:
            if sha256(f)!=info['image_sha256'][Path(f).name]:raise ValueError('Image changed since preprocessing')
        self.windows=json.loads((folder/'filter_windows.json').read_text())
        self.patches=np.load(folder/'query_patches.npy',mmap_mode='r',allow_pickle=False)
        if self.patches.shape!=(len(frames),24,768) or self.patches.dtype!=np.float32:
            raise ValueError('Wrong query patch shape or precision')
        if not np.isfinite(self.patches).all():raise ValueError('Nonfinite cached patches')
        self.query_patch_cache={f:self.patches[i] for i,f in enumerate(frames)}
        self.provenance={'root':str(root.resolve()),'manifest_sha256':sha256(root/'manifest.json'),
                         'scan_manifest_sha256':sha256(path),'scan':sid,'gpu_required':False}

    def kept_images(self,wi,original):
        row=self.windows[wi]
        if row['window']!=wi or row['original_images']!=[Path(f).name for f in original]:
            raise ValueError('Window differs from preprocessing')
        byname={Path(f).name:f for f in original}
        kept=[byname[n] for n in row['kept_images']]
        if len(set(kept))!=len(kept):raise ValueError('Repeated kept images in cache')
        return kept
