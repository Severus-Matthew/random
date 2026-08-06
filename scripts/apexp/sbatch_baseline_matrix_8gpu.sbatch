#!/usr/bin/env bash
#SBATCH --job-name=apexp_baseline_matrix
#SBATCH --partition=ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/week2/baseline_matrix_cap200_all_methods_all_k/slurm_%j.out
#SBATCH --error=/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/week2/baseline_matrix_cap200_all_methods_all_k/slurm_%j.err

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD

source /fsx/jmanvi/Internship_project/.venv/bin/activate

export LARGE_ROOT="results/apexp_block/config_large_500"
export BASELINE_ROOT="$LARGE_ROOT/week2/baseline_matrix_cap200_all_methods_all_k"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

export N_GPUS=8
export SEED=42
export FORCE="${FORCE:-0}"

echo "job id: $SLURM_JOB_ID"
echo "node: $(hostname)"
nvidia-smi

bash scripts/apexp/run_baseline_matrix_8gpu.sh "$BASELINE_ROOT/baseline_jobs.tsv"
