# SGR³: Scene Graph Retrieval and Reasoning

This directory contains the SGR³ pipeline for scene graph generation from RGB sequences. It uses ColQwen filtering, retrieval with SigLIP2 patches, and local Qwen3-VL-32B-Instruct inference through vLLM. Reference scans from the same physical scenes as the evaluation scans are excluded using the official 3RScan reference/rescan groups.

## Configuration

Default parameters are provided in [configs/paper.json](configs/paper.json). See the paper for the method and experimental settings. Before running, resolve the configuration and set the paths to your data, model, index, and feature caches as shown below. Retrieval caches must match the configuration used to create them.

## Entry Points

| Script | Purpose |
|---|---|
| `tools/prepare_config.py` | Resolve paths and validate the configuration |
| `tools/audit_kb.py` | Check that the knowledge base excludes evaluation scenes |
| `tools/precompute_retrieval.py` | Prepare the retrieval cache |
| `tools/run_local.py` | Run generation, object matching, and evaluation |
| `tools/evaluate.py` | Evaluate existing predictions |
| `tools/preprocess.py` | Prepare query features and filtering decisions |
| `tools/embed_reference.py` | Compute reference embeddings |
| `tools/build_kb.py` | Build the knowledge base index |

Inference uses a local model and does not require a DashScope API key.

## Running the Pipeline

Set the package, Python environment, and model paths for your installation. Replace the example paths below. `requirements-observed.txt` lists versions from the existing environment; it is not a complete installation lock file. Use an environment with the dependencies required by the entry point, including FAISS and PyTorch for retrieval.

```bash
SGR3_PACKAGE='/path/to/SGR3Model/sgr3'
SGR3_PYTHON='/path/to/environment/bin/python'
QWEN_MODEL_PATH='/path/to/models/Qwen3-VL-32B-Instruct'
export SGR3_PACKAGE SGR3_PYTHON QWEN_MODEL_PATH
```

Create a resolved configuration. The default project root is the parent of `sgr3`. Use `--project-root` to point to the project containing your data and cached artifacts, and check the paths in the generated configuration.

```bash
"$SGR3_PYTHON" "$SGR3_PACKAGE/tools/prepare_config.py" --project-root /path/to/project --output /path/to/work/paper_config.json

"$SGR3_PYTHON" "$SGR3_PACKAGE/tools/audit_kb.py" --config /path/to/work/paper_config.json --actual-keys

"$SGR3_PYTHON" "$SGR3_PACKAGE/tools/run_local.py" --config /path/to/work/paper_config.json
```

The generation, preprocessing, embedding, index construction, and retrieval preparation commands print plans or perform preflight checks unless `--execute` is supplied. If preflight reports `needs_new_k10_retrieval_cache`, prepare the retrieval cache first.

Prepare the retrieval cache, then run a single scan:

```bash
/path/to/faiss-environment/bin/python "$SGR3_PACKAGE/tools/precompute_retrieval.py" --config /path/to/work/paper_config.json --execute

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

"$SGR3_PYTHON" "$SGR3_PACKAGE/tools/run_local.py" --config /path/to/work/paper_config.json --model-path "$QWEN_MODEL_PATH" --output-root /path/to/work/pilot --scans SCAN_UUID --execute
```

Omit `--scans` to process all 47 scans. Use a new output directory for each run. Scans are processed sequentially; the entry point does not submit Slurm jobs or resume an interrupted output directory. Longer runs can process separate scan subsets, whose predictions can then be combined in one evaluation manifest.

Generation reads images and reference edges from the prepared retrieval cache without loading FAISS. Each output directory stores the configuration, source snapshots, model path, window prompts and responses, object mappings, final graphs, and scores. `slurm/local_qwen_example.sbatch` provides an example job script.

## Evaluation

Protocol: `robio-category-equivalence-directed-set-v4`.

Object equivalences are defined in `evaluation/object_aliases_v2.json` and applied consistently to both endpoint categories. Examples include chair/armchair, table/desk, and heater/radiator. Predicate judging by a semantic model is disabled.

1. GT IDs and predicted IDs must exist in their respective graphs. GT to prediction mappings must be one to one.
2. For each directed object pair, merge predicates in exported order, normalize aliases, remove duplicates, exclude none candidates, and retain the first three.
3. `H` counts positive GT relationship annotations whose endpoints are associated with objects of equivalent categories and whose predicate appears among the retained candidates. Multiple annotations for one pair are counted separately.
4. `D_Old` counts positive GT annotations with correctly categorized, associated endpoints and a nonempty predicted candidate set. `Old = H / D_Old`.
5. `D_New` counts all positive GT relationships in the fixed evaluation split: 10,125 for all 47 scans. `New = H / D_New`. Unmatched objects, incorrect categories, omitted pairs, and failed predictions remain in this denominator.
6. `New mRecall` averages `H_r / D_New,r` equally across the 25 predicate classes with positive GT support, including classes with zero hits. Counts are pooled across scans before calculating recall.
7. `object_category_recall_after_association` is the number of correctly categorized, associated GT objects divided by all GT objects. It differs from classifier Object R@10 and from instance association recall without a category check.
8. `node_ratio` is the total number of predicted nodes divided by the total number of GT nodes.

Relationship set@3 differs from joint triplet R@1. Native classifier Object R@10 and Predicate R@3 are separate metrics. Graphs from different methods can use this shared evaluator after conversion to the prediction and mapping formats below. GNN export uses the top category and predicates ranked by score after thresholding.

Example prediction:

```json
{"nodes":[{"id":10,"name":"armchair"},{"id":20,"name":"desk"}],"edges":[{"src":10,"tgt":20,"c_edge":["left","close by","front"]}]}
```

Example mapping, from GT ID to predicted ID:

```json
{"1":"10","2":"20","3":"none"}
```

A wrapper of the form `{"gt2pred":{...}}` is also accepted. Convert legacy JSONL matching records into one mapping file per scan before evaluation.

An evaluation manifest references the full split and the available scan outputs:

```json
{
  "method": "SGR3_paper",
  "split": "/path/to/SGR3Model/sgr3/splits/evaluation_scans.txt",
  "GT_root": "/path/to/project/data/3RScan",
  "scans": {
    "SCAN_UUID": {
      "state": "complete",
      "prediction": "/path/to/graph_final.json",
      "mapping": "/path/to/object_matching.json"
    }
  }
}
```

Scans missing from the manifest are marked pending, and their GT annotations remain in the New denominator. Outputs with pending scans are marked as partial results.

```bash
python "$SGR3_PACKAGE/tools/evaluate.py" --manifest /path/to/manifest.json --output /path/to/work/scores.json
```

The generation entry point writes manifests and scores automatically. Read the shared evaluation results from `reports/manifest_XX_scores.json`. The matching output `runs/.../object_matching/results.json` contains separate legacy diagnostics.

## Data and Cache Dependencies

The repository contains code, the evaluation split, taxonomies, and object aliases. Models, RGB images, GT data, embeddings, FAISS indexes, and feature caches must be supplied separately. Configure their locations in the resolved configuration.

The default configuration references an existing KB with scene exclusion and completed ColQwen/SigLIP query caches. Index construction excludes all evaluation scene families even if older embeddings for those scans are present in the embedding directory.

Optional preprocessing also requires the existing `siglip2_val/patch_tokens_fp16.npy` and `paths.json` files for each scan as proxies for selecting comparison candidates. The default workflow reuses the prepared caches.

To recompute reference features or construct an index, use `tools/embed_reference.py` or `tools/build_kb.py` with a new output directory, then update the configuration to reference the new artifacts. To recompute query features, use `tools/preprocess.py` and finish all 47 scans before pointing `precomputed_cache_dir` to the new cache. A single shard is not a complete cache.
