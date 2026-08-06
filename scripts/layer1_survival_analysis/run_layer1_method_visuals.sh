#!/usr/bin/env bash
set -euo pipefail
LAYER1_DIR="${1:-results/apex_layer1}"
OUT_DIR="${2:-$LAYER1_DIR/method_clean}"
TEMP="${3:-0}"
python src/phase3_apex/layer1_method_visuals.py \
  --layer1-dir "$LAYER1_DIR" \
  --output-dir "$OUT_DIR" \
  --temperature "$TEMP"
