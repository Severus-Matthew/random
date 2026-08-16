#!/bin/bash
#SBATCH --job-name=apex_quality_balanced_k1_to_16_survival_utility_
#SBATCH --output=slurm_logs/paper_eval_cap50_profile_kgrid/apex/apex_quality_balanced_k1_to_16_survival_utility_controller_v_%j.out
#SBATCH --error=slurm_logs/paper_eval_cap50_profile_kgrid/apex/apex_quality_balanced_k1_to_16_survival_utility_controller_v_%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:6
#SBATCH --exclusive
#SBATCH --cpus-per-task=48

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
export APEXP_LEARNED_FAST_TRACE_SCORES=0

export APEXP_FAST_CANDIDATE_KS="2,4,8,12,16"
export APEXP_MIN_K=2
export APEXP_MAX_K=16

export APEXP_LEARNED_FAST_ASYNC=1
export APEXP_LEARNED_FAST_EVERY_N=4
export APEXP_LEARNED_FAST_FIRST_WINDOW=3
export APEXP_FAST_WINDOW=3
export APEXP_FAST_MIN_OBS=3
export APEXP_ONLINE_SCORE_UTILITY_WEIGHT=0.7
export APEXP_ONLINE_SCORE_ACCEPT_WEIGHT=0.22
export APEXP_ONLINE_SCORE_ACCEPT_RATE_WEIGHT=0.75
export APEXP_ONLINE_SCORE_TPS_WEIGHT=0.25
export APEXP_ONLINE_SCORE_WASTE_WEIGHT=0.80
export APEXP_ONLINE_SCORE_BLOCK_PENALTY_WEIGHT=0.12
export APEXP_ONLINE_MIN_EXPECTED_ACCEPTED=0.80
export APEXP_ONLINE_MIN_ACCEPT_RATE=0.12
export APEXP_ONLINE_BAD_ACTION_PENALTY=4.0

export OUT_DIR="results/apexp_block/config_large_500/evaluation/paper_cap50_profile_kgrid/apex_hot_runs/survival_utility_controller_v4b_rankgroup_speed_lr5e4_rank2__quality_balanced__k2_4_8_12_16"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "[APEX PROFILE-KGRID JOB]"
echo "run_name=survival_utility_controller_v4b_rankgroup_speed_lr5e4_rank2__quality_balanced__k2_4_8_12_16"
echo "model_variant=survival_utility_controller_v4b_rankgroup_speed_lr5e4_rank2"
echo "profile=quality_balanced"
echo "candidate_set=k2_4_8_12_16"
echo "candidate_ks=2,4,8,12,16"
echo "min_k=2"
echo "out_dir=$OUT_DIR"
which nvcc || true
nvcc --version || true

python3 src/apexp/week2/realtime_hot_e2e.py \
  --slow-router-dir "results/apexp_block/config_large_500/week2/request_prompt_router_d128" \
  --out-dir "$OUT_DIR" \
  --max-prompts-per-file 0 \
  --fast-min-k 2 \
  --fast-candidate-ks "2,4,8,12,16" \
  --fast-window 3 \
  --fast-min-obs 3 \
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
  --gpu-memory-utilization 0.75 \
  2>&1 | tee "$OUT_DIR/run.log"
