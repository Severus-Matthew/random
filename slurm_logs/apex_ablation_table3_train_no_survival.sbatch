#!/bin/bash
#SBATCH --job-name=apex_no_surv_train
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm_logs/apex_ablation_table3_train_no_survival_%j.out
#SBATCH --error=slurm_logs/apex_ablation_table3_train_no_survival_%j.err

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

DATA="results/apexp_block/config_large_500/week2/online_block_dataset_core.csv"
BASE_ROOT="results/apexp_block/config_large_500/week2/survival_utility_controller_v4"
OUT_ROOT="results/apexp_block/config_large_500/week2/ablation_no_survival_loss_v4"

# Reuse prompt_text_map from the existing balanced metadata.
PROMPT_TEXT_MAP=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("results/apexp_block/config_large_500/week2/survival_utility_controller_v4/balanced/metadata.json")
if not p.exists():
    raise SystemExit("missing metadata: " + str(p))
m = json.loads(p.read_text())
print(m["prompt_text_map"])
PY
)

echo "DATA=$DATA"
echo "PROMPT_TEXT_MAP=$PROMPT_TEXT_MAP"
echo "OUT_ROOT=$OUT_ROOT"

mkdir -p "$OUT_ROOT"

run_train () {
  NAME="$1"
  AA="$2"
  AW="$3"
  AG="$4"
  AM="$5"

  OUT="$OUT_ROOT/$NAME"
  rm -rf "$OUT"
  mkdir -p "$OUT"

  echo
  echo "============================================================"
  echo "[TRAIN no-survival] $NAME"
  echo "alpha_accept=$AA alpha_waste=$AW alpha_global=$AG alpha_method=$AM"
  echo "============================================================"

  python3 src/apexp/week2/train_survival_utility_controller_v4.py \
    --data "$DATA" \
    --prompt-text-map "$PROMPT_TEXT_MAP" \
    --out-dir "$OUT" \
    --epochs 12 \
    --batch-size 16384 \
    --seed 42 \
    --hidden 512 \
    --dropout 0.10 \
    --lr 1e-3 \
    --prompt-emb-dim 64 \
    --alpha-accept "$AA" \
    --alpha-waste "$AW" \
    --alpha-global "$AG" \
    --alpha-method "$AM" \
    --lambda-survival 0.0 \
    --lambda-cost 0.20 \
    --lambda-tps 0.30 \
    --lambda-utility 1.00 \
    --lambda-rank 1.00 \
    --grouped-rank-batches \
    2>&1 | tee "$OUT/train.log"
}

# Match the original v4 operating objective weights.
run_train "APEX_B_no_survival_loss" 0.30 0.35 0.15 0.20
run_train "APEX_S_no_survival_loss" 0.20 0.20 0.20 0.15
run_train "APEX_E_no_survival_loss" 0.40 0.50 0.10 0.25

echo "[DONE] no-survival training"
