#!/bin/bash
#SBATCH --job-name=apex_eval_survival_utility_controller_v4_balance
#SBATCH --output=slurm_logs/evaluation_jobs/apex_eval_survival_utility_controller_v4_balanced__temp0p1___%j.out
#SBATCH --error=slurm_logs/evaluation_jobs/apex_eval_survival_utility_controller_v4_balanced__temp0p1___%j.err
#SBATCH --partition=ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=420G
 

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export APEXP_ONLINE_FAST=1
export APEXP_LEARNED_FAST_MODEL_DIR="results/apexp_block/config_large_500/week2/survival_utility_controller_v4/balanced"
export APEXP_LEARNED_FAST_DEVICE=cpu
export APEXP_LEARNED_FAST_EVERY_N=1
export APEXP_LEARNED_FAST_SWITCH_MARGIN=0.03
export APEXP_LEARNED_FAST_TRACE_SCORES=0

export APEXP_ONLINE_FAST_TRACE=1
export APEXP_FAST_CANDIDATE_KS="4,8,16"
export APEXP_MIN_K=4
export APEXP_MAX_K=16
export APEXP_FAST_WINDOW=2
export APEXP_FAST_MIN_OBS=2

export OUT_DIR="results/apexp_block/config_large_500/evaluation/hot_runs/survival_utility_controller_v4_balanced__temp0p1__k4_8_16"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 src/apexp/week2/realtime_hot_e2e.py   --slow-router-dir "results/apexp_block/config_large_500/week2/request_prompt_router_d128"   --out-dir "$OUT_DIR"   --max-prompts-per-file 20   --fast-candidate-ks "4,8,16"   --fast-window 2   --fast-min-obs 2   --slow-gpus "ngram_sd:0,eagle3:1,draft_sd:2"   --fast-gpus "ngram_sd:3,eagle3:4,draft_sd:5"   --target-model "Qwen/Qwen3-8B"   --draft-model "Qwen/Qwen2.5-1.5B-Instruct"   --eagle3-model "RedHatAI/Qwen3-8B-speculator.eagle3"   --input-jsonl     data/benchmarks/By_split_phase_1/code_gen_test.jsonl \
    data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl \
    data/benchmarks/By_split_phase_1/long_context_completion_test.jsonl \
    data/benchmarks/By_split_phase_1/long_chain_reasoning_test.jsonl \
    data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl \
    data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl \
    data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl \
    data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl   --gpu-memory-utilization 0.7   2>&1 | tee "$OUT_DIR/run.log"
