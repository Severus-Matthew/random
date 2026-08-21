#!/bin/bash
#SBATCH --job-name=apex_hot_learned20
#SBATCH --output=slurm_logs/apex_hot_learned20_%j.out
#SBATCH --error=slurm_logs/apex_hot_learned20_%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export APEXP_LEARNED_FAST_DEVICE=cpu

export LARGE_ROOT="results/apexp_block/config_large_500"
export SLOW_DIR="$LARGE_ROOT/week2/request_prompt_router_d128"
export MODEL_DIR="$LARGE_ROOT/week2/survival_utility_controller_v4/speed_first"
export OUT_DIR="$LARGE_ROOT/week2/realtime_hot_e2e_learned_speed_first_cap20"

export APEXP_ONLINE_FAST=1
export APEXP_LEARNED_FAST_MODEL_DIR="$MODEL_DIR"
export APEXP_ONLINE_FAST_TRACE=1
export APEXP_FAST_CANDIDATE_KS="1,2,4,8,16"
export APEXP_MIN_K=1
export APEXP_MAX_K=16
export APEXP_FAST_WINDOW=2
export APEXP_FAST_MIN_OBS=2

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 src/apexp/week2/realtime_hot_e2e.py \
  --slow-router-dir "$SLOW_DIR" \
  --out-dir "$OUT_DIR" \
  --max-prompts-per-file 20 \
  --fast-candidate-ks "1,2,4,8,16" \
  --fast-window 2 \
  --fast-min-obs 2 \
  --slow-gpus "ngram_sd:0,eagle3:1,draft_sd:2" \
  --fast-gpus "ngram_sd:3,eagle3:4,draft_sd:5" \
  --input-jsonl \
    data/benchmarks/By_split_phase_1/code_gen_test.jsonl \
    data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl \
    data/benchmarks/By_split_phase_1/long_context_completion_test.jsonl \
  --gpu-memory-utilization 0.70 \
  2>&1 | tee "$OUT_DIR/run.log"
