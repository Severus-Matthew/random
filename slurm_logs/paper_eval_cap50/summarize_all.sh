#!/bin/bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python3 evaluation/03_summarize_hot_runs.py \
  --runs-root results/apexp_block/config_large_500/evaluation/paper_cap50/apex_hot_runs \
  --out-dir results/apexp_block/config_large_500/evaluation/paper_cap50/apex_hot_summary \
  --force

python3 evaluation/summarize_fixed_baseline_cap50.py \
  --root results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs \
  --out-dir results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_summary

echo
echo "APEX summary:"
echo "  results/apexp_block/config_large_500/evaluation/paper_cap50/apex_hot_summary"
echo
echo "Baseline summary:"
echo "  results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_summary"
