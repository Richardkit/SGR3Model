# SGR³: Scene Graph Retrieval and Reasoning

SGR³ generates 3D scene graphs from RGB sequences using retrieved scene graph context and local Qwen3-VL inference. See the paper for the method and evaluation definitions.

## Setup

Default settings are in [configs/paper.json](configs/paper.json). Set the paths to your data, model, index, and feature caches. Dependency versions are listed in [requirements-observed.txt](requirements-observed.txt); install the packages needed for your environment.

Models, RGB images, ground truth annotations, and retrieval artifacts are supplied separately. Reuse compatible caches or prepare them with `tools/preprocess.py`, `tools/embed_reference.py`, and `tools/build_kb.py`. Query preprocessing requires each scan's `siglip2_val/patch_tokens_fp16.npy` and `paths.json`, and the cache must cover the full evaluation split.

## Entry Points

| Script | Purpose |
|---|---|
| `tools/prepare_config.py` | Resolve paths and validate settings |
| `tools/audit_kb.py` | Check knowledge base scene exclusion |
| `tools/precompute_retrieval.py` | Prepare retrieval results |
| `tools/run_local.py` | Generate graphs, match objects, and evaluate |
| `tools/evaluate.py` | Score existing predictions |

## Usage

From the repository root, prepare a configuration, build the retrieval cache, and run the pipeline. Replace the example paths with your own.

```bash
python sgr3/tools/prepare_config.py --project-root /path/to/project --output /path/to/work/config.json
python sgr3/tools/precompute_retrieval.py --config /path/to/work/config.json --execute
python sgr3/tools/run_local.py --config /path/to/work/config.json --model-path /path/to/Qwen3-VL-32B-Instruct --output-root /path/to/work/run --execute
```

Use a new output directory. Add `--scans SCAN_UUID` to run a subset; otherwise all scans in the split are processed. Commands that support `--execute` perform preflight checks or print a plan when it is omitted. Run any entry point with `--help` for its options.

## Evaluation

Evaluation reports relationship set@3 recall and mean recall using fixed category equivalences. The pipeline creates an evaluation manifest containing the split, ground truth paths, predictions, and GT to prediction object mappings.

To score an existing manifest:

```bash
python sgr3/tools/evaluate.py --manifest /path/to/manifest.json --output /path/to/scores.json
```

Pipeline scores are saved in `reports/manifest_XX_scores.json` inside the output directory.
