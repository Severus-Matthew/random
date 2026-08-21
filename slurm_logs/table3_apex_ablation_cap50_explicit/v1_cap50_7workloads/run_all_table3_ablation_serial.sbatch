#!/bin/bash
#SBATCH --job-name=table3_apex_ablation_v1_cap50_7workloads
#SBATCH --partition=ml-p6-b200-48xlarge-us-west-2a-tp-p6-agent-cust-2026-q3-n8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output=slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/run_all_table3_ablation_serial.out
#SBATCH --error=slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/run_all_table3_ablation_serial.err

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi || true

SCRIPTS=(slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_B__quality_balanced__k1_to_16__base.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_B__quality_balanced__k1_to_16__no_survival_loss.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_B__quality_balanced__k1_to_16__no_cost_head.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_B__quality_balanced__k1_to_16__no_entropy_features.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_B__quality_balanced__k1_to_16__no_repetition_features.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_B__quality_balanced__k1_to_16__no_verifier_history_features.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_S__quality_balanced__k1_to_16__base.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_S__quality_balanced__k1_to_16__no_survival_loss.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_S__quality_balanced__k1_to_16__no_cost_head.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_S__quality_balanced__k1_to_16__no_entropy_features.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_S__quality_balanced__k1_to_16__no_repetition_features.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_S__quality_balanced__k1_to_16__no_verifier_history_features.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_E__quality_balanced__k1_to_16__base.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_E__quality_balanced__k1_to_16__no_survival_loss.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_E__quality_balanced__k1_to_16__no_cost_head.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_E__quality_balanced__k1_to_16__no_entropy_features.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_E__quality_balanced__k1_to_16__no_repetition_features.sbatch slurm_logs/table3_apex_ablation_cap50_explicit/v1_cap50_7workloads/variant_scripts/APEX_E__quality_balanced__k1_to_16__no_verifier_history_features.sbatch)

for s in "${SCRIPTS[@]}"; do
  echo
  echo "============================================================"
  echo "[START] $(date) $s"
  echo "============================================================"
  bash "$s"
  echo "============================================================"
  echo "[DONE]  $(date) $s"
  echo "============================================================"
done

echo "[ALL DONE] $(date)"
