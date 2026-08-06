#!/bin/bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD

SCRIPT="slurm_logs/run_hot_ablation_fixed_candidateks.sbatch"

if [ ! -f "$SCRIPT" ]; then
  echo "Missing $SCRIPT"
  exit 1
fi

echo "Submitting KFIX ablation jobs..."

# 1. window=2, all k = 1..16
sbatch "$SCRIPT" \
  cap50_w2_allk_1to16_KFIX \
  2 2 1 16 \
  "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" \
  50

# 2. window=1, all k = 1..16
sbatch "$SCRIPT" \
  cap50_w1_allk_1to16_KFIX \
  1 1 1 16 \
  "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" \
  50

# 3. window=1, sparse/intermediate k
sbatch "$SCRIPT" \
  cap50_w1_ks_1_2_3_4_5_6_8_12_16_KFIX \
  1 1 1 16 \
  "1,2,3,4,5,6,8,12,16" \
  50

# 4. window=2, sparse/intermediate k
sbatch "$SCRIPT" \
  cap50_w2_ks_1_2_3_4_5_6_8_12_16_KFIX \
  2 2 1 16 \
  "1,2,3,4,5,6,8,12,16" \
  50

# 5. window=2, min_k=2, sparse/intermediate k
sbatch "$SCRIPT" \
  cap50_w2_mink2_ks_2_3_4_5_6_8_12_16_KFIX \
  2 2 2 16 \
  "2,3,4,5,6,8,12,16" \
  50

echo
echo "Submitted. Current queue:"
squeue -u "$USER"
