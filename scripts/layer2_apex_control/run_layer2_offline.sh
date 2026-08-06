#!/usr/bin/env bash
set -euo pipefail

LAYER1_DIR="${1:-results/apex_layer1}"
OUT_DIR="${2:-results/apex_layer2}"
TEMP_FILTER="${3:-0}"

INPUT="${LAYER1_DIR}/layer1_request_survival.csv"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing input: $INPUT"
  echo "Run Layer 1 first:"
  echo "  bash scripts/phase3_apex/run_layer1_existing_results.sh results results/apex_layer1 16"
  exit 1
fi

mkdir -p "$OUT_DIR"

python src/phase3_apex/layer2_apex_control.py \
  --input "$INPUT" \
  --output-dir "$OUT_DIR" \
  --temperature "$TEMP_FILTER" \
  --seeds 0,1,2 \
  --train-frac 0.7 \
  --ucb-alpha 1.0 \
  --lambda-tail 0.0

echo
echo "Layer 2 outputs:"
echo "  $OUT_DIR/policy_decisions.csv"
echo "  $OUT_DIR/controller_comparison.csv"
echo "  $OUT_DIR/controller_by_workload.csv"
echo "  $OUT_DIR/figures/"
