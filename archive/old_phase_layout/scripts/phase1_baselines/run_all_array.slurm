#!/usr/bin/env bash
#SBATCH --job-name=spec_all
#SBATCH --output=logs/spec_all_%A_%a.out
#SBATCH --error=logs/spec_all_%A_%a.err
#SBATCH --partition=${PARTITION:-ghx4}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:${GPUS:-1}
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-6

set -euo pipefail
mkdir -p logs

# NOTE: This array job runs each workload as a separate task.
# The per-workload slurm files (slurm/run_*.slurm) contain #SBATCH headers
# that are only parsed by the scheduler at sbatch submission time.
# When called via `bash` here (inside a running job), those headers are IGNORED —
# which is correct and intentional: this parent job already allocated the resources.
# All env vars (TARGET_MODEL, DRAFT_MODEL, etc.) must be set before submitting this array job,
# or use `export` in your submission wrapper.

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export EAGLE3_MODEL="${EAGLE3_MODEL:-}"
export TP="${TP:-1}"
export LIMIT="${LIMIT:-50}"
export SEED="${SEED:-42}"

WORKLOADS=(
  code_gen
  long_horizon_swe
  hardware_gen
  mathematical_reasoning
  long_chain_reasoning
  long_context_completion
  conversational_generation
)

# Per-workload MAX_TOKENS overrides (long workloads need more room)
declare -A WORKLOAD_MAX_TOKENS
WORKLOAD_MAX_TOKENS["code_gen"]=256
WORKLOAD_MAX_TOKENS["long_horizon_swe"]=1024
WORKLOAD_MAX_TOKENS["hardware_gen"]=256
WORKLOAD_MAX_TOKENS["mathematical_reasoning"]=512
WORKLOAD_MAX_TOKENS["long_chain_reasoning"]=1024
WORKLOAD_MAX_TOKENS["long_context_completion"]=2048
WORKLOAD_MAX_TOKENS["conversational_generation"]=256

WORKLOAD=${WORKLOADS[$SLURM_ARRAY_TASK_ID]}
export MAX_TOKENS="${WORKLOAD_MAX_TOKENS[$WORKLOAD]}"

echo "[ARRAY] Task $SLURM_ARRAY_TASK_ID → workload=$WORKLOAD max_tokens=$MAX_TOKENS"

# Source the workload script directly (its #SBATCH lines are no-ops here, that's fine)
bash "slurm/run_${WORKLOAD}.slurm"
