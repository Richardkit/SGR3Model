import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = PACKAGE.parent


def load_config(path=None, project_root=None):
    path = Path(path) if path else PACKAGE/'configs/paper.json'
    project = Path(project_root).resolve() if project_root else DEFAULT_PROJECT
    def expand(value):
        if isinstance(value, str):
            value = value.replace('${PROJECT_ROOT}', str(project)).replace('${PACKAGE_ROOT}', str(PACKAGE))
            if '${' in value: raise ValueError('Unresolved config path placeholder')
            return value
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {expand(key): expand(item) for key, item in value.items()}
        return value

    cfg = expand(json.loads(path.read_text(encoding='utf-8')))
    expected = {'window_size':6,'stride':4,'filter_warmup_windows':2,'colqwen_threshold':0.5,
                'retrieval_temperature':0.5,'patch_topk':10,'reference_frame_topk':5,
                'retrieval_selection_policy':'global_frame_sum_topk_v1',
                'retrieval_exclusion_policy':'scene','generation_image_policy':'original_window'}
    for key,value in expected.items():
        if cfg.get(key)!=value: raise ValueError(f'Paper configuration mismatch: {key} must be {value!r}')
    if cfg['api']['temperature']!=0 or cfg['api']['enable_thinking'] is not False:
        raise ValueError('Local generation requires temperature 0 and thinking disabled')
    if cfg['api']['base_url']!='local://vllm':raise ValueError('This package uses local Qwen, no remote API')
    return cfg


def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

