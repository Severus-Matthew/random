#!/bin/bash
#SBATCH --job-name=ar_long_context_completion
#SBATCH --partition=ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/fsx/jmanvi/Internship_project/ASD/slurm_logs/paper_eval_final_full6/ar/ar_long_context_completion_%j.out
#SBATCH --error=/fsx/jmanvi/Internship_project/ASD/slurm_logs/paper_eval_final_full6/ar/ar_long_context_completion_%j.err

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

mkdir -p "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs/long_context_completion/ar/ar_temp0.0"

echo "[AR FINAL6]"
echo "workload=long_context_completion"
echo "input_jsonl=/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion_test.jsonl"
echo "out_dir=/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs/long_context_completion/ar/ar_temp0.0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi || true

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "long_context_completion" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion_test.jsonl" \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs/long_context_completion/ar/ar_temp0.0" \
  --experiment "paper_eval_final_full6_ar" \
  --run_name "long_context_completion_full" \
  --method ar \
  --k 1 \
  --limit 0 \
  --temperature 0.0 \
  --max_tokens 2048 \
  --seed 42 \
  --enforce_eager \
  --gpu_memory_utilization 0.80 \
  --model "Qwen/Qwen3-8B"

echo "[done] long_context_completion"
