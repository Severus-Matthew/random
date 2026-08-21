#!/bin/bash
#SBATCH --job-name=apex_survival_utility_controller_v4b_rankgroup_s
#SBATCH --output=slurm_logs/paper_eval_cap50_async4_min3/apex/apex_survival_utility_controller_v4b_rankgroup_speed_lr5e4_r_%j.out
#SBATCH --error=slurm_logs/paper_eval_cap50_async4_min3/apex/apex_survival_utility_controller_v4b_rankgroup_speed_lr5e4_r_%j.err
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_HOME=/usr/local/cuda-12.9
export CUDA_PATH=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export NVCC=/usr/local/cuda-12.9/bin/nvcc

export APEXP_ONLINE_FAST=1
export APEXP_LEARNED_FAST_MODEL_DIR="results/apexp_block/config_large_500/week2/survival_utility_controller_v4b_rankgroup/speed_lr5e4_rank2"
export APEXP_LEARNED_FAST_DEVICE=cpu
export APEXP_LEARNED_FAST_EVERY_N=4
export APEXP_LEARNED_FAST_SWITCH_MARGIN=0.03
export APEXP_LEARNED_FAST_TRACE_SCORES=0
export APEXP_ONLINE_BAD_ACTION_PENALTY=4.0
export APEXP_ONLINE_MIN_ACCEPT_RATE=0.08
export APEXP_ONLINE_MIN_EXPECTED_ACCEPTED=1.25
export APEXP_ONLINE_SCORE_BLOCK_PENALTY_WEIGHT=0.25
export APEXP_ONLINE_SCORE_WASTE_WEIGHT=0.35
export APEXP_ONLINE_SCORE_TPS_WEIGHT=0.45
export APEXP_ONLINE_SCORE_ACCEPT_RATE_WEIGHT=0.30
export APEXP_ONLINE_SCORE_ACCEPT_WEIGHT=0.12
export APEXP_ONLINE_SCORE_UTILITY_WEIGHT=1.0
export APEXP_LEARNED_FAST_FIRST_WINDOW=3
export APEXP_LEARNED_FAST_ASYNC=1

export APEXP_ONLINE_FAST_TRACE=1
export APEXP_FAST_CANDIDATE_KS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
export APEXP_MIN_K=3
export APEXP_MAX_K=16
export APEXP_FAST_WINDOW=3
export APEXP_FAST_MIN_OBS=3

export OUT_DIR="results/apexp_block/config_large_500/evaluation/paper_cap50_async4_min3/apex_hot_runs/survival_utility_controller_v4b_rankgroup_speed_lr5e4_rank2__k1_to_16"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "[APEX JOB] run_name=survival_utility_controller_v4b_rankgroup_speed_lr5e4_rank2__k1_to_16"
echo "[APEX JOB] model=results/apexp_block/config_large_500/week2/survival_utility_controller_v4b_rankgroup/speed_lr5e4_rank2"
echo "[APEX JOB] candidate_ks=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
echo "[APEX JOB] out_dir=$OUT_DIR"
which nvcc || true
nvcc --version || true

python3 src/apexp/week2/realtime_hot_e2e.py \
  --slow-router-dir "results/apexp_block/config_large_500/week2/request_prompt_router_d128" \
  --out-dir "$OUT_DIR" \
  --max-prompts-per-file 0 \
  --fast-min-k 3 \
  --fast-candidate-ks "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" \
  --fast-window 2 \
  --fast-min-obs 2 \
  --slow-gpus "ngram_sd:0,eagle3:1,draft_sd:2" \
  --fast-gpus "ngram_sd:3,eagle3:4,draft_sd:5" \
  --target-model "Qwen/Qwen3-8B" \
  --draft-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3-model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --input-jsonl \
    data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/code_gen_cap50.jsonl \
    data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl \
    data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_context_completion_cap50.jsonl \
    data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_chain_reasoning_cap50.jsonl \
    data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_horizon_swe_cap50.jsonl \
    data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/hardware_gen_cap50.jsonl \
    data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_sft_cap50.jsonl \
    data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_gen_cap50.jsonl \
  --gpu-memory-utilization 0.50 \
  2>&1 | tee "$OUT_DIR/run.log"
