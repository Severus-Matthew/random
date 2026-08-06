#!/bin/bash
#SBATCH --job-name=apex_fast_v4_3gpu
#SBATCH --partition=c # Partition (queue) name
#SBATCH --output=slurm_logs/apex_fast_v4_3gpu_%j.out
#SBATCH --error=slurm_logs/apex_fast_v4_3gpu_%j.err
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=48:00:00

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
export OUT_ROOT="$LARGE_ROOT/week2/survival_utility_controller_v4"

mkdir -p "$OUT_ROOT"
mkdir -p slurm_logs

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "Start time: $(date)"
echo "ONLINE_DATA=$ONLINE_DATA"
echo "PROMPT_MAP=$PROMPT_MAP"
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

  local out="$OUT_ROOT/$name"
  rm -rf "$out"
  mkdir -p "$out"

  echo
  echo "========================================================================"
  echo "START TRAINING: $name on local GPU $gpu"
  echo "alpha_accept=$alpha_accept"
  echo "alpha_waste=$alpha_waste"
  echo "alpha_global=$alpha_global"
  echo "alpha_method=$alpha_method"
  echo "out=$out"
  echo "========================================================================"

  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=12 \
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

  echo "DONE TRAINING: $name on GPU $gpu"
}

train_one 0 speed_first 0.20 0.20 0.20 0.15 &
PID_SPEED=$!

train_one 1 balanced 0.30 0.35 0.15 0.20 &
PID_BALANCED=$!

train_one 2 efficient 0.40 0.50 0.10 0.25 &
PID_EFFICIENT=$!

echo "Launched:"
echo "speed_first PID=$PID_SPEED"
echo "balanced    PID=$PID_BALANCED"
echo "efficient   PID=$PID_EFFICIENT"

wait "$PID_SPEED"
wait "$PID_BALANCED"
wait "$PID_EFFICIENT"

echo
echo "DONE. Models:"
find "$OUT_ROOT" -name "model.pt" -print

echo
echo "Preprocessors:"
find "$OUT_ROOT" -name "preprocessor.pkl" -print

echo
echo "Prompt embedders:"
find "$OUT_ROOT" -name "prompt_embedder.pkl" -print

echo
echo "Metadata:"
find "$OUT_ROOT" -name "metadata.json" -print

echo
echo "Training histories:"
find "$OUT_ROOT" -name "training_history.csv" -print

echo
echo "End time: $(date)"
