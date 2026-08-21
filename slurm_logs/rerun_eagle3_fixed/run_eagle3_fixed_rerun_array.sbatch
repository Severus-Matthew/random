#!/bin/bash
#SBATCH --job-name=eagle3_fixed_rerun
#SBATCH --partition=ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-34%4
#SBATCH --output=/fsx/jmanvi/Internship_project/ASD/slurm_logs/rerun_eagle3_fixed/eagle3_fixed_rerun_%A_%a.out
#SBATCH --error=/fsx/jmanvi/Internship_project/ASD/slurm_logs/rerun_eagle3_fixed/eagle3_fixed_rerun_%A_%a.err

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

# Use cached models. Comment these out if the model needs downloading.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi || true

python3 evaluation/rerun_eagle3_fixed_baselines.py \
  --mode run-one \
  --manifest "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/final_evaluations/eagle3_rerun_manifest.jsonl" \
  --index "${SLURM_ARRAY_TASK_ID}"
