#!/bin/bash
#SBATCH --job-name=apex_fast_v4b4
#SBATCH --output=slurm_logs/apex_fast_v4b4_%j.out
#SBATCH --error=slurm_logs/apex_fast_v4b4_%j.err
#SBATCH --partition=dev
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=64

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12

export LARGE_ROOT="results/apexp_block/config_large_500"
export ONLINE_DATA="$LARGE_ROOT/week2/online_block_dataset_core.csv"
export PROMPT_MAP="$LARGE_ROOT/week2/prompt_text_map.csv"
export OUT_ROOT="$LARGE_ROOT/week2/survival_utility_controller_v4b_rankgroup"

mkdir -p "$OUT_ROOT"
mkdir -p slurm_logs

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "Start time: $(date)"
echo "ONLINE_DATA=$ONLINE_DATA"
echo "OUT_ROOT=$OUT_ROOT"
echo "============================================================"

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

if [ ! -f "$ONLINE_DATA" ]; then
  echo "Missing $ONLINE_DATA"
  exit 1
fi

if [ ! -f "$PROMPT_MAP" ]; then
  echo "Missing $PROMPT_MAP"
  exit 1
fi

python3 -m py_compile src/apexp/week2/train_survival_utility_controller_v4.py

train_one () {
  local gpu="$1"
  local name="$2"
  local alpha_accept="$3"
  local alpha_waste="$4"
  local alpha_global="$5"
  local alpha_method="$6"
  local lr="$7"
  local lambda_rank="$8"
  local lambda_utility="$9"

  local out="$OUT_ROOT/$name"
  rm -rf "$out"
  mkdir -p "$out"

  echo
  echo "========================================================================"
  echo "START TRAINING: $name on GPU $gpu"
  echo "alpha_accept=$alpha_accept alpha_waste=$alpha_waste alpha_global=$alpha_global alpha_method=$alpha_method"
  echo "lr=$lr lambda_rank=$lambda_rank lambda_utility=$lambda_utility"
  echo "out=$out"
  echo "========================================================================"

  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=12 \
  python3 src/apexp/week2/train_survival_utility_controller_v4.py \
    --data "$ONLINE_DATA" \
    --prompt-text-map "$PROMPT_MAP" \
    --out-dir "$out" \
    --epochs 30 \
    --batch-size 5000 \
    --max-rows 0 \
    --hidden 469 \
    --dropout 0.05 \
    --lr "$lr" \
    --prompt-emb-dim 128 \
    --alpha-accept "$alpha_accept" \
    --alpha-waste "$alpha_waste" \
    --alpha-global "$alpha_global" \
    --alpha-method "$alpha_method" \
    --lambda-survival 1.00 \
    --lambda-cost 0.15 \
    --lambda-tps 0.50 \
    --lambda-utility "$lambda_utility" \
    --lambda-rank "$lambda_rank" \
    --grouped-rank-batches \
    2>&1 | tee "$out/train.log"

  echo "DONE TRAINING: $name on GPU $gpu"
}

# Best previous objective, tuned around speed.
train_one 0 speed_lr5e4_rank2 0.20 0.20 0.20 0.15 5e-4 2.0 1.5 &
PID_A=$!

# Same speed objective, lower LR and stronger ranking.
train_one 1 speed_lr3e4_rank3 0.20 0.20 0.20 0.15 3e-4 3.0 1.5 &
PID_B=$!

# Balanced objective for paper-quality speed/waste tradeoff.
train_one 2 balanced_lr5e4_rank2 0.30 0.35 0.15 0.20 5e-4 2.0 1.5 &
PID_C=$!

# Efficient/waste-sensitive objective for Pareto-frontier comparison.
train_one 3 efficient_lr5e4_rank2 0.40 0.50 0.10 0.25 5e-4 2.0 1.5 &
PID_D=$!

echo "Launched:"
echo "speed_lr5e4_rank2      PID=$PID_A"
echo "speed_lr3e4_rank3      PID=$PID_B"
echo "balanced_lr5e4_rank2   PID=$PID_C"
echo "efficient_lr5e4_rank2  PID=$PID_D"

wait "$PID_A"
wait "$PID_B"
wait "$PID_C"
wait "$PID_D"

echo
echo "DONE. Models:"
find "$OUT_ROOT" -name "model.pt" -print

echo
echo "Training histories:"
find "$OUT_ROOT" -name "training_history.csv" -print

echo
echo "End time: $(date)"
