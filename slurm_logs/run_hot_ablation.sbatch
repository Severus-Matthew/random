#!/bin/bash
#SBATCH --job-name=apex_hot_ablate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:6
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=dev # Partition (queue) name



set -euo pipefail

CONFIG_NAME="$1"
WINDOW="$2"
MIN_OBS="$3"
MIN_K="$4"
MAX_K="$5"
CANDIDATE_KS="$6"
MAX_PROMPTS_PER_FILE="${7:-50}"

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export LARGE_ROOT="results/apexp_block/config_large_500"
export SLOW_DIR="$LARGE_ROOT/week2/request_prompt_router_d128"
export OUT_DIR="$LARGE_ROOT/week2/realtime_hot_e2e_${CONFIG_NAME}_tracefix"

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export APEXP_ONLINE_FAST=1
export APEXP_ONLINE_FAST_TRACE=1
export APEXP_FAST_WINDOW="$WINDOW"
export APEXP_FAST_MIN_OBS="$MIN_OBS"
export APEXP_MIN_K="$MIN_K"
export APEXP_MAX_K="$MAX_K"
export APEXP_FAST_CANDIDATE_KS="$CANDIDATE_KS"

echo "CONFIG_NAME=$CONFIG_NAME"
echo "OUT_DIR=$OUT_DIR"
echo "APEXP_FAST_WINDOW=$APEXP_FAST_WINDOW"
echo "APEXP_FAST_MIN_OBS=$APEXP_FAST_MIN_OBS"
echo "APEXP_MIN_K=$APEXP_MIN_K"
echo "APEXP_MAX_K=$APEXP_MAX_K"
echo "APEXP_FAST_CANDIDATE_KS=$APEXP_FAST_CANDIDATE_KS"
echo "MAX_PROMPTS_PER_FILE=$MAX_PROMPTS_PER_FILE"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 src/apexp/week2/realtime_hot_e2e.py \
  --slow-router-dir "$SLOW_DIR" \
  --out-dir "$OUT_DIR" \
  --max-prompts-per-file "$MAX_PROMPTS_PER_FILE" \
  --slow-gpus "ngram_sd:0,eagle3:1,draft_sd:2" \
  --fast-gpus "ngram_sd:3,eagle3:4,draft_sd:5" \
  --input-jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion_synthetic.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning_synthetic.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_train.jsonl \
  2>&1 | tee "$OUT_DIR/run.log"

echo "DONE: $CONFIG_NAME"
