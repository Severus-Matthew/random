#!/bin/bash
set -euo pipefail
cd /fsx/jmanvi/Internship_project/ASD
for j in slurm_logs/paper_eval_cap50_profile_kgrid/ar/*.sbatch; do
  echo "Submitting $j"
  sbatch "$j"
  sleep 15
done
