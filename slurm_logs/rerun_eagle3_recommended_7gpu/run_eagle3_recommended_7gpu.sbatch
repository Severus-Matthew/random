#!/bin/bash
#SBATCH --job-name=eagle3_recommended_7gpu
#SBATCH --partition=ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=700G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/rerun_eagle3_recommended_7gpu/eagle3_recommended_7gpu_%j.out
#SBATCH --error=slurm_logs/rerun_eagle3_recommended_7gpu/eagle3_recommended_7gpu_%j.err

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export APEXP_RERUN_GPU_MEM=0.60

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi || true

python3 evaluation/rerun_eagle3_recommended_7gpu_by_workload.py
