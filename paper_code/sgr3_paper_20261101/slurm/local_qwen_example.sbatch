#!/bin/bash
#SBATCH --account=hk-project-p0025551
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=sgr3-paper-globalframes
# Template only. Not submitted by this package. Pass CONFIG OUTPUT [--scans UUID ...].
set -euo pipefail
: "${SGR3_PACKAGE:?Set the absolute package path}"
: "${SGR3_PYTHON:?Set the tested Python executable}"
: "${QWEN_MODEL_PATH:?Set the local Qwen3-VL-32B-Instruct checkpoint path}"
config_path="${1:?Pass the resolved configuration}"
result_path="${2:?Pass a new output directory}"
shift 2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec "$SGR3_PYTHON" -u "$SGR3_PACKAGE/tools/run_local.py" --config "$config_path" --model-path "$QWEN_MODEL_PATH" --output-root "$result_path" "$@" --execute
