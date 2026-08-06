#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-results/apex_layer1/layer1_request_survival.csv}"
OUT_DIR="${2:-results/apex_layer2_stress_2c}"
TEMP="${3:-0}"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing input: $INPUT"
  echo "Expected Layer 1 request survival CSV."
  exit 1
fi

mkdir -p "$OUT_DIR"

python src/layer2_apex_control/layer2_stress_and_2c.py \
  --input "$INPUT" \
  --output-dir "$OUT_DIR" \
  --temperature "$TEMP" \
  --seeds 0,1,2 \
  --train-frac 0.7 \
  --ucb-alpha 1.0 \
  --bootstrap 2000

echo
echo "Layer 2 stress + 2C-lite complete."
echo "Main outputs:"
echo "  $OUT_DIR/controller_comparison_common_support.csv"
echo "  $OUT_DIR/paired_bootstrap_ci.csv"
echo "  $OUT_DIR/per_workload_apex_vs_ucb_bestfixed.csv"
echo "  $OUT_DIR/selected_method_k_distribution.csv"
echo "  $OUT_DIR/ablation_summary.csv"
echo "  $OUT_DIR/hazard_2c_lite_calibration.csv"
echo "  $OUT_DIR/figures/"
