#!/bin/bash
#SBATCH --job-name=t3_apex_s_base
#SBATCH --partition=ml-p6-b200-48xlarge-us-west-2a-tp-p6-agent-cust-2026-q3-n8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_S__quality_balanced__k1_to_16__base.out
#SBATCH --error=slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_S__quality_balanced__k1_to_16__base.err

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

export OUT_DIR="results/apexp_block/config_large_500/evaluation/table3_apex_ablation_cap50_explicit_v1_cap50_7workloads/apex_hot_runs/APEX_S__quality_balanced__k1_to_16__base"
export APEXP_ONLINE_FAST=1
export APEXP_LEARNED_FAST_MODEL_DIR="results/apexp_block/config_large_500/week2/survival_utility_controller_v4/speed_first"
export APEXP_LEARNED_FAST_DEVICE="cuda"
export APEXP_LEARNED_FAST_TRACE_SCORES=0

export APEXP_TABLE3_ANCHOR="APEX_S"
export APEXP_TABLE3_PAPER_ANCHOR="APEX-S"
export APEXP_TABLE3_ABLATION="base"
export APEXP_TABLE3_VARIANT="Full APEX"

export APEXP_FAST_CANDIDATE_KS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
export APEXP_MIN_K=1
export APEXP_MAX_K=16

export APEXP_ABLATE_ENTROPY="0"
export APEXP_ABLATE_REPETITION="0"
export APEXP_ABLATE_VERIFIER_HISTORY="0"
export APEXP_FAST_MIN_OBS="3"
export APEXP_FAST_WINDOW="3"
export APEXP_LEARNED_FAST_ASYNC="1"
export APEXP_LEARNED_FAST_EVERY_N="4"
export APEXP_LEARNED_FAST_FIRST_WINDOW="3"
export APEXP_ONLINE_BAD_ACTION_PENALTY="4.0"
export APEXP_ONLINE_MIN_ACCEPT_RATE="0.12"
export APEXP_ONLINE_MIN_EXPECTED_ACCEPTED="0.80"
export APEXP_ONLINE_SCORE_ACCEPT_RATE_WEIGHT="0.75"
export APEXP_ONLINE_SCORE_ACCEPT_WEIGHT="0.22"
export APEXP_ONLINE_SCORE_BLOCK_PENALTY_WEIGHT="0.12"
export APEXP_ONLINE_SCORE_TPS_WEIGHT="0.25"
export APEXP_ONLINE_SCORE_UTILITY_WEIGHT="0.7"
export APEXP_ONLINE_SCORE_WASTE_WEIGHT="0.80"

echo "============================================================"
echo "TABLE 3 APEX ABLATION"
echo "RUN_NAME=APEX_S__quality_balanced__k1_to_16__base"
echo "OUT_DIR=$OUT_DIR"
echo "ANCHOR=$APEXP_TABLE3_ANCHOR"
echo "VARIANT=$APEXP_TABLE3_VARIANT"
echo "MODEL_DIR=$APEXP_LEARNED_FAST_MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "============================================================"
env | grep -E 'APEXP_(TABLE3|ONLINE_SCORE|ONLINE_MIN|LEARNED_FAST|FAST_|ABLATE)' | sort
nvidia-smi || true

if [ -e "$OUT_DIR" ]; then
  echo "[ERROR] OUT_DIR already exists. Refusing to overwrite for reproducibility:"
  echo "$OUT_DIR"
  echo "Move/delete it manually only if you intentionally want to rerun this exact config."
  exit 44
fi

mkdir -p "$OUT_DIR"

python3 src/apexp/week2/realtime_hot_e2e.py \
  --slow-router-dir "results/apexp_block/config_large_500/week2/request_prompt_router_d128" \
  --out-dir "$OUT_DIR" \
  --max-prompts-per-file 0 \
  --fast-candidate-ks "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" \
  --fast-min-k 1 \
  --fast-window 3 \
  --fast-min-obs 3 \
  --slow-gpus "ngram_sd:0,eagle3:1,draft_sd:2" \
  --fast-gpus "ngram_sd:3,eagle3:4,draft_sd:5" \
  --target-model "Qwen/Qwen3-8B" \
  --draft-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3-model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --input-jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/code_gen_cap50.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_sft_cap50.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_gen_cap50.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_chain_reasoning_cap50.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_context_completion_cap50.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_horizon_swe_cap50.jsonl \
    /fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl \
  --gpu-memory-utilization 0.60 \
  2>&1 | tee "$OUT_DIR/run.log"
