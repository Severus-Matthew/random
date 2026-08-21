#!/bin/bash
#SBATCH --job-name=base_draft_sd_k2
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
#SBATCH --output=/fsx/jmanvi/Internship_project/ASD/slurm_logs/paper_eval_final_full6/baselines/base_draft_sd_k2_%j.out
#SBATCH --error=/fsx/jmanvi/Internship_project/ASD/slurm_logs/paper_eval_final_full6/baselines/base_draft_sd_k2_%j.err

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

echo "[BASELINE FINAL6] method=draft_sd k=2"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi || true

echo "[BASELINE FINAL6] method=draft_sd k=2 workload=code_gen"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "paper_eval_final_full6_fixed" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 0 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "draft_sd" \
  --k 2 \
  --max_tokens 512 \
  --temperature 0.0 \
  --seed 42 \
  --enforce_eager \
  --gpu_memory_utilization 0.60

echo "[BASELINE FINAL6] method=draft_sd k=2 workload=conversational_generation"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "conversational_generation" \
  --run_name "conversational_generation" \
  --experiment "paper_eval_final_full6_fixed" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test.jsonl" \
  --limit 0 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "draft_sd" \
  --k 2 \
  --max_tokens 512 \
  --temperature 0.0 \
  --seed 42 \
  --enforce_eager \
  --gpu_memory_utilization 0.60

echo "[BASELINE FINAL6] method=draft_sd k=2 workload=long_chain_reasoning"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "paper_eval_final_full6_fixed" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning_test.jsonl" \
  --limit 0 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "draft_sd" \
  --k 2 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --seed 42 \
  --enforce_eager \
  --gpu_memory_utilization 0.60

echo "[BASELINE FINAL6] method=draft_sd k=2 workload=long_context_completion"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "paper_eval_final_full6_fixed" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion_test.jsonl" \
  --limit 0 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "draft_sd" \
  --k 2 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --seed 42 \
  --enforce_eager \
  --gpu_memory_utilization 0.60

echo "[BASELINE FINAL6] method=draft_sd k=2 workload=long_horizon_swe"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "paper_eval_final_full6_fixed" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 0 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "draft_sd" \
  --k 2 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --seed 42 \
  --enforce_eager \
  --gpu_memory_utilization 0.60

echo "[BASELINE FINAL6] method=draft_sd k=2 workload=mathematical_reasoning"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "paper_eval_final_full6_fixed" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 0 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6/fixed_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "draft_sd" \
  --k 2 \
  --max_tokens 512 \
  --temperature 0.0 \
  --seed 42 \
  --enforce_eager \
  --gpu_memory_utilization 0.60

