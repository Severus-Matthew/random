#!/usr/bin/env bash
set -euo pipefail
RESULTS_DIR="${1:-results}"
OUT_DIR="${2:-results/apex_layer1}"
MAX_DEPTH="${3:-16}"
python src/phase3_apex/layer1_survival_analysis.py \
  --all-runs "${RESULTS_DIR}/all_runs.csv" \
  --out-dir "${OUT_DIR}" \
  --max-depth "${MAX_DEPTH}"
