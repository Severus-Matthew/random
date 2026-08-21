#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs results

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export EAGLE3_MODEL="${EAGLE3_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
export TP="${TP:-1}"
export LIMIT="${LIMIT:-50}"
export MAX_TOKENS="${MAX_TOKENS:-1024}"
export SEED="${SEED:-42}"

export RUN_NAME="long_horizon_swe"
export INPUT_JSONL="/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl"
WORKLOAD="long_horizon_swe"

if [[ -z "${EAGLE3_MODEL:-}" ]]; then
  echo "[ERROR] EAGLE3_MODEL is empty. Because DEFAULT_METHOD=eagle3, set EAGLE3_MODEL before running."
  exit 2
fi

# Eagle3 is tried first by default.
bash /fsx/jmanvi/Internship_project/ASD/scripts/phase1_baselines/run_one_workload.sh "$WORKLOAD" eagle3 4

# Core baselines.
bash /fsx/jmanvi/Internship_project/ASD/scripts/phase1_baselines/run_one_workload.sh "$WORKLOAD" ar 1
bash /fsx/jmanvi/Internship_project/ASD/scripts/phase1_baselines/run_one_workload.sh "$WORKLOAD" ngram_sd 4
bash /fsx/jmanvi/Internship_project/ASD/scripts/phase1_baselines/run_one_workload.sh "$WORKLOAD" draft_sd 4

# Fixed-k sensitivity sweep.
for K in 1 2 8 16; do
  bash /fsx/jmanvi/Internship_project/ASD/scripts/phase1_baselines/run_one_workload.sh "$WORKLOAD" eagle3 "$K"
  bash /fsx/jmanvi/Internship_project/ASD/scripts/phase1_baselines/run_one_workload.sh "$WORKLOAD" ngram_sd "$K"
  bash /fsx/jmanvi/Internship_project/ASD/scripts/phase1_baselines/run_one_workload.sh "$WORKLOAD" draft_sd "$K"
done
