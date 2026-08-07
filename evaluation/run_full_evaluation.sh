#!/bin/bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

LARGE_ROOT="${LARGE_ROOT:-results/apexp_block/config_large_500}"
EVAL_ROOT="$LARGE_ROOT/evaluation"

mkdir -p "$EVAL_ROOT"

python3 evaluation/01_plot_training.py \
  --roots \
    "$LARGE_ROOT/week2/survival_utility_controller_v4" \
    "$LARGE_ROOT/week2/survival_utility_controller_v4b_rankgroup" \
  --out-dir "$EVAL_ROOT/training_study"

python3 evaluation/02_make_hot_eval_jobs.py \
  --out-dir slurm_logs/evaluation_jobs \
  --result-root "$EVAL_ROOT/hot_runs" \
  --slow-router-dir "$LARGE_ROOT/week2/request_prompt_router_d128" \
  --model-dirs \
    "$LARGE_ROOT/week2/survival_utility_controller_v4/speed_first" \
    "$LARGE_ROOT/week2/survival_utility_controller_v4/balanced" \
    "$LARGE_ROOT/week2/survival_utility_controller_v4/efficient" \
    "$LARGE_ROOT/week2/survival_utility_controller_v4b_rankgroup/speed_lr5e4_rank2" \
  --workload-files \
    data/benchmarks/By_split_phase_1/code_gen_test.jsonl \
    data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl \
    data/benchmarks/By_split_phase_1/long_context_completion_test.jsonl \
    data/benchmarks/By_split_phase_1/long_chain_reasoning_test.jsonl \
    data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl \
    data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl \
    data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl \
    data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl \
  --temps 0.0 0.1 0.5 \
  --candidate-sets \
    k8_16=8,16 \
    k4_8_16=4,8,16 \
    k1_16=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --max-prompts-per-file "${MAX_PROMPTS_PER_FILE:-20}"

python3 evaluation/04_collect_fixed_baselines.py \
  --roots "$LARGE_ROOT" \
  --out-dir "$EVAL_ROOT/fixed_baselines"

python3 evaluation/05_check_official_baselines.py \
  --out-dir "$EVAL_ROOT/external_baselines"

echo
echo "Generated evaluation assets under: $EVAL_ROOT"
echo "Generated Slurm jobs under: slurm_logs/evaluation_jobs"
echo "Submit with: bash slurm_logs/evaluation_jobs/submit_all.sh"
echo "After jobs finish, run:"
echo "  python3 evaluation/03_summarize_hot_runs.py --runs-root $EVAL_ROOT/hot_runs --out-dir $EVAL_ROOT/hot_summary"
