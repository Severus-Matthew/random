#!/bin/bash
#SBATCH --job-name=ar_mathematical_reasoning
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/paper_eval_cap50_profile_kgrid/ar/ar_mathematical_reasoning.out
#SBATCH --error=slurm_logs/paper_eval_cap50_profile_kgrid/ar/ar_mathematical_reasoning.err

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_HOME=/usr/local/cuda-12.9
export CUDA_PATH=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export NVCC=/usr/local/cuda-12.9/bin/nvcc

# Important: do NOT overwrite CUDA_VISIBLE_DEVICES here.
# Slurm should decide which physical GPU this one-GPU AR job receives.

mkdir -p "results/apexp_block/config_large_500/evaluation/paper_cap50_profile_kgrid/fixed_runs/mathematical_reasoning/ar/ar_temp0.0"

echo "[AR CAP50 JOB]"
echo "workload=mathematical_reasoning"
echo "input_jsonl=data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl"
echo "out_dir=results/apexp_block/config_large_500/evaluation/paper_cap50_profile_kgrid/fixed_runs/mathematical_reasoning/ar/ar_temp0.0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl" \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50_profile_kgrid/fixed_runs/mathematical_reasoning/ar/ar_temp0.0" \
  --experiment "paper_cap50_ar" \
  --run_name "mathematical_reasoning_full" \
  --method ar \
  --k 1 \
  --limit 0 \
  --temperature 0.0 \
  --max_tokens 512 \
  --seed 42 \
  --enforce_eager \
  --gpu_memory_utilization 0.8 \
  --model "Qwen/Qwen3-8B"

echo "[done] mathematical_reasoning"
