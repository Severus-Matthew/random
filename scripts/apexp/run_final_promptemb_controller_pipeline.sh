#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-results/apexp_block/config_full}"
GPU_ID="${2:-6}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

DATA="$ROOT/week2/online_block_dataset.csv"
PROMPT_MAP="$ROOT/week2/prompt_text_map.csv"
AUDIT="$ROOT/week2/prompt_text_map_audit.csv"
BENCH="data/benchmarks/By_split_phase_1"
OUT="$ROOT/week2/neural_controller_promptemb_final"

mkdir -p "$OUT/logs"

echo "ROOT=$ROOT"
echo "DATA=$DATA"
echo "PROMPT_MAP=$PROMPT_MAP"
echo "OUT=$OUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: Missing $DATA"
  exit 1
fi

echo "=== Build prompt text map ==="
python src/apexp/week2/build_prompt_text_map.py \
  --data "$DATA" \
  --benchmark-dir "$BENCH" \
  --out "$PROMPT_MAP" \
  --audit-out "$AUDIT" \
  --min-match-rate 0.90 \
  2>&1 | tee "$OUT/logs/build_prompt_text_map_$(date +%Y%m%d_%H%M%S).log"

echo "=== Train full survival-cost controller ==="
python src/apexp/week2/train_neural_survival_controller.py \
  --data "$DATA" \
  --prompt-text-map "$PROMPT_MAP" \
  --out-dir "$OUT/full" \
  --variant full \
  --epochs 20 \
  --batch-size 2048 \
  --lr 1e-3 \
  --max-rows 1500000 \
  --lambda-cost 0.25 \
  2>&1 | tee "$OUT/logs/train_full_$(date +%Y%m%d_%H%M%S).log"

echo "=== Train no-cost ablation ==="
python src/apexp/week2/train_neural_survival_controller.py \
  --data "$DATA" \
  --prompt-text-map "$PROMPT_MAP" \
  --out-dir "$OUT/no_cost" \
  --variant no_cost \
  --epochs 20 \
  --batch-size 2048 \
  --lr 1e-3 \
  --max-rows 1500000 \
  2>&1 | tee "$OUT/logs/train_no_cost_$(date +%Y%m%d_%H%M%S).log"

echo "=== Train direct/no-survival ablation ==="
python src/apexp/week2/train_neural_survival_controller.py \
  --data "$DATA" \
  --prompt-text-map "$PROMPT_MAP" \
  --out-dir "$OUT/direct" \
  --variant direct \
  --epochs 20 \
  --batch-size 2048 \
  --lr 1e-3 \
  --max-rows 1500000 \
  2>&1 | tee "$OUT/logs/train_direct_$(date +%Y%m%d_%H%M%S).log"

echo "=== Offline replay on held-out prompt split ==="
python src/apexp/week2/evaluate_neural_controller_replay.py \
  --data "$DATA" \
  --prompt-text-map "$PROMPT_MAP" \
  --out-dir "$OUT/replay" \
  --full-model "$OUT/full" \
  --no-cost-model "$OUT/no_cost" \
  --direct-model "$OUT/direct" \
  2>&1 | tee "$OUT/logs/replay_$(date +%Y%m%d_%H%M%S).log"

echo "=== Verify no workload input ==="
python - <<PY
import json
from pathlib import Path

for name in ["full", "no_cost", "direct"]:
    p = Path("$OUT") / name / "metadata.json"
    meta = json.loads(p.read_text())
    print("\\n", name)
    print("uses_workload_as_input:", meta["uses_workload_as_input"])
    print("uses_prompt_embedding:", meta["uses_prompt_embedding"])
    print("cat_cols:", meta["cat_cols"])
    assert meta["uses_workload_as_input"] is False
    assert "workload" not in meta["cat_cols"]
PY

echo "=== DONE ==="
echo "Main outputs:"
echo "$OUT/full/metadata.json"
echo "$OUT/full/prompt_embedder.pkl"
echo "$OUT/full/preprocessor.pkl"
echo "$OUT/full/model.pt"
echo "$OUT/full/training_history.csv"
echo "$OUT/replay/policy_summary.csv"
echo "$OUT/replay/apexp_full_action_distribution.csv"
