#!/bin/bash
#SBATCH --job-name=apex_base_ngram_sd_k8
#SBATCH --output=slurm_logs/paper_eval_cap50/baselines/apex_base_ngram_sd_k8_%j.out
#SBATCH --error=slurm_logs/paper_eval_cap50/baselines/apex_base_ngram_sd_k8_%j.err
#SBATCH --partition=ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

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

echo "[BASELINE JOB] method=ngram_sd k=8"
echo "[BASELINE JOB] out_root=results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs"
which nvcc || true
nvcc --version || true


echo "[BASELINE] method=ngram_sd k=8 workload=code_gen"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "fixed_baseline_cap50" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/code_gen_cap50.jsonl" \
  --limit 0 \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "ngram_sd" \
  --k 8 \
  --max_tokens 512 \
  --temperature 0.0 \
  --gpu_memory_utilization 0.50 \
  

echo "[BASELINE] method=ngram_sd k=8 workload=mathematical_reasoning"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "fixed_baseline_cap50" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl" \
  --limit 0 \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "ngram_sd" \
  --k 8 \
  --max_tokens 512 \
  --temperature 0.0 \
  --gpu_memory_utilization 0.50 \
  

echo "[BASELINE] method=ngram_sd k=8 workload=long_context_completion"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "fixed_baseline_cap50" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_context_completion_cap50.jsonl" \
  --limit 0 \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "ngram_sd" \
  --k 8 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --gpu_memory_utilization 0.50 \
  

echo "[BASELINE] method=ngram_sd k=8 workload=long_chain_reasoning"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "fixed_baseline_cap50" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_chain_reasoning_cap50.jsonl" \
  --limit 0 \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "ngram_sd" \
  --k 8 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --gpu_memory_utilization 0.50 \
  

echo "[BASELINE] method=ngram_sd k=8 workload=long_horizon_swe"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "fixed_baseline_cap50" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_horizon_swe_cap50.jsonl" \
  --limit 0 \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "ngram_sd" \
  --k 8 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --gpu_memory_utilization 0.50 \
  

echo "[BASELINE] method=ngram_sd k=8 workload=hardware_gen"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "fixed_baseline_cap50" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/hardware_gen_cap50.jsonl" \
  --limit 0 \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "ngram_sd" \
  --k 8 \
  --max_tokens 512 \
  --temperature 0.0 \
  --gpu_memory_utilization 0.50 \
  

echo "[BASELINE] method=ngram_sd k=8 workload=conversational_generation_sft"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "fixed_baseline_cap50" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_sft_cap50.jsonl" \
  --limit 0 \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "ngram_sd" \
  --k 8 \
  --max_tokens 512 \
  --temperature 0.0 \
  --gpu_memory_utilization 0.50 \
  

echo "[BASELINE] method=ngram_sd k=8 workload=conversational_generation_gen"

python3 src/layer0_data_collection/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "fixed_baseline_cap50" \
  --input_jsonl "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_gen_cap50.jsonl" \
  --limit 0 \
  --out_dir "results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \
  --method "ngram_sd" \
  --k 8 \
  --max_tokens 512 \
  --temperature 0.0 \
  --gpu_memory_utilization 0.50 \
  

