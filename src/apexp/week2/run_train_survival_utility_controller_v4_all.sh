#!/bin/bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD

export LARGE_ROOT="results/apexp_block/config_large_500"
export ONLINE_DATA="$LARGE_ROOT/week2/online_block_dataset_core.csv"
export PROMPT_MAP="$LARGE_ROOT/week2/prompt_text_map.csv"
export OUT_ROOT="$LARGE_ROOT/week2/survival_utility_controller_v4"

mkdir -p "$OUT_ROOT"

if [ ! -f "$ONLINE_DATA" ]; then
  echo "Missing $ONLINE_DATA"
  echo "Run make_online_block_dataset.py first."
  exit 1
fi

run_variant () {
  local name="$1"
  local alpha_accept="$2"
  local alpha_waste="$3"
  local alpha_global="$4"
  local alpha_method="$5"

  local out="$OUT_ROOT/$name"
  rm -rf "$out"
  mkdir -p "$out"

  echo
  echo "========================================================================"
  echo "TRAIN SURVIVAL-UTILITY CONTROLLER: $name"
  echo "alpha_accept=$alpha_accept alpha_waste=$alpha_waste alpha_global=$alpha_global alpha_method=$alpha_method"
  echo "========================================================================"

  python3 src/apexp/week2/train_survival_utility_controller_v4.py \
    --data "$ONLINE_DATA" \
    --prompt-text-map "$PROMPT_MAP" \
    --out-dir "$out" \
    --epochs 12 \
    --batch-size 16384 \
    --max-rows 0 \
    --hidden 512 \
    --dropout 0.10 \
    --lr 1e-3 \
    --prompt-emb-dim 64 \
    --alpha-accept "$alpha_accept" \
    --alpha-waste "$alpha_waste" \
    --alpha-global "$alpha_global" \
    --alpha-method "$alpha_method" \
    --lambda-survival 1.00 \
    --lambda-cost 0.20 \
    --lambda-tps 0.30 \
    --lambda-utility 1.00 \
    --lambda-rank 1.00 \
    2>&1 | tee "$out/train.log"
}

# Most speed-oriented: best first attempt for beating fixed Eagle3 speed.
run_variant speed_first 0.20 0.20 0.20 0.15

# Main paper-quality tradeoff: speed + accepted progress + waste.
run_variant balanced 0.30 0.35 0.15 0.20

# More conservative/efficiency-oriented.
run_variant efficient 0.40 0.50 0.10 0.25

echo
echo "DONE. Models:"
find "$OUT_ROOT" -name "model.pt" -print

echo
echo "Training histories:"
find "$OUT_ROOT" -name "training_history.csv" -print
