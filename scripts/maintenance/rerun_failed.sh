#!/usr/bin/env bash
set -e

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_chain_reasoning/draft_sd/k4_temp0.0_ctx256_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 256 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_chain_reasoning/draft_sd/k4_temp0.0_ctx128_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 128 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_chain_reasoning/draft_sd/k4_temp0.0_ctx64_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 64 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_chain_reasoning/draft_sd/k4_temp0.0_ctx1024_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 1024 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_chain_reasoning/draft_sd/k4_temp0.0_ctx512_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 512 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_sft/draft_sd/k4_temp0.0_ctx64_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 64 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_sft/draft_sd/k4_temp0.0_ctx512_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 512 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_sft/draft_sd/k4_temp0.0_ctx1024_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 1024 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_sft/draft_sd/k4_temp0.0_ctx128_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 128 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_sft/draft_sd/k4_temp0.0_ctx256_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 256 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/hardware_gen/draft_sd/k4_temp0.0_ctx1024_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 1024 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/hardware_gen/draft_sd/k4_temp0.0_ctx512_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 512 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/hardware_gen/draft_sd/k4_temp0.0_ctx256_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 256 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/hardware_gen/draft_sd/k4_temp0.0_ctx128_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 128 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/hardware_gen/draft_sd/k4_temp0.0_ctx64_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 64 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_context_completion/draft_sd/k4_temp0.0_ctx256_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 256 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_context_completion/draft_sd/k4_temp0.0_ctx128_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 128 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_context_completion/draft_sd/k4_temp0.0_ctx1024_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 1024 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_context_completion/draft_sd/k4_temp0.0_ctx64_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 64 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_context_completion/draft_sd/k4_temp0.0_ctx512_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 512 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/code_gen/draft_sd/k4_temp0.0_ctx256_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 256 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/code_gen/draft_sd/k4_temp0.0_ctx128_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 128 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/code_gen/draft_sd/k4_temp0.0_ctx64_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 64 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/code_gen/draft_sd/k4_temp0.0_ctx1024_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 1024 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/code_gen/draft_sd/k4_temp0.0_ctx512_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 512 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_horizon_swe/draft_sd/k4_temp0.0_ctx64_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 64 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_horizon_swe/draft_sd/k4_temp0.0_ctx256_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 256 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_horizon_swe/draft_sd/k4_temp0.0_ctx128_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 128 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_horizon_swe/draft_sd/k4_temp0.0_ctx512_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 512 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/long_horizon_swe/draft_sd/k4_temp0.0_ctx1024_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 1024 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/mathematical_reasoning/draft_sd/k4_temp0.0_ctx64_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 64 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/mathematical_reasoning/draft_sd/k4_temp0.0_ctx512_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 512 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/mathematical_reasoning/draft_sd/k4_temp0.0_ctx1024_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 1024 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/mathematical_reasoning/draft_sd/k4_temp0.0_ctx128_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 128 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/mathematical_reasoning/draft_sd/k4_temp0.0_ctx256_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 256 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_gen/draft_sd/k4_temp0.0_ctx512_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 512 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_gen/draft_sd/k4_temp0.0_ctx1024_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 1024 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_gen/draft_sd/k4_temp0.0_ctx64_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 64 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_gen/draft_sd/k4_temp0.0_ctx128_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 128 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/F_context_growth/conversational_generation_gen/draft_sd/k4_temp0.0_ctx256_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "F_context_growth" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 256 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/conversational_generation_gen/draft_sd/k2_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 2 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/conversational_generation_gen/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/conversational_generation_gen/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/hardware_gen/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/hardware_gen/draft_sd/k2_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 2 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/hardware_gen/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_horizon_swe/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_horizon_swe/draft_sd/k2_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 2 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_horizon_swe/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/code_gen/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/code_gen/draft_sd/k2_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 2 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/conversational_generation_sft/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/conversational_generation_sft/draft_sd/k2_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 2 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/conversational_generation_sft/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/mathematical_reasoning/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/mathematical_reasoning/draft_sd/k2_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 2 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/mathematical_reasoning/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_chain_reasoning/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_chain_reasoning/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_chain_reasoning/draft_sd/k2_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 2 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_context_completion/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_context_completion/draft_sd/k2_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 2 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/K_rejection_study/long_context_completion/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "K_rejection_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/G_multi_turn/long_horizon_swe/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "G_multi_turn" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/G_multi_turn/long_horizon_swe/draft_sd/k4_temp0.0_t8_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "G_multi_turn" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 8 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/G_multi_turn/long_horizon_swe/draft_sd/k4_temp0.0_t4_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "G_multi_turn" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 4 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/G_multi_turn/long_horizon_swe/draft_sd/k4_temp0.0_t2_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "G_multi_turn" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 2 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/G_multi_turn/code_gen/draft_sd/k4_temp0.0_t8_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "G_multi_turn" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 8 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/G_multi_turn/code_gen/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "G_multi_turn" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/G_multi_turn/code_gen/draft_sd/k4_temp0.0_t2_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "G_multi_turn" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 2 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/G_multi_turn/code_gen/draft_sd/k4_temp0.0_t4_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "G_multi_turn" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 4 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/I_entropy_bucket_study/long_chain_reasoning/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "I_entropy_bucket_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/I_entropy_bucket_study/conversational_generation_gen/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_gen" \
  --run_name "conversational_generation_gen" \
  --experiment "I_entropy_bucket_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/I_entropy_bucket_study/code_gen/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "I_entropy_bucket_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/I_entropy_bucket_study/long_horizon_swe/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "I_entropy_bucket_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/I_entropy_bucket_study/hardware_gen/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "hardware_gen" \
  --run_name "hardware_gen" \
  --experiment "I_entropy_bucket_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/I_entropy_bucket_study/mathematical_reasoning/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "I_entropy_bucket_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/I_entropy_bucket_study/conversational_generation_sft/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "conversational_generation_sft" \
  --run_name "conversational_generation_sft" \
  --experiment "I_entropy_bucket_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/I_entropy_bucket_study/long_context_completion/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "I_entropy_bucket_study" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/long_horizon_swe/draft_sd/k4_temp1.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/long_horizon_swe/draft_sd/k4_temp0.3_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.3 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/long_horizon_swe/draft_sd/k4_temp0.6_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_horizon_swe" \
  --run_name "long_horizon_swe" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.6 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/mathematical_reasoning/draft_sd/k4_temp1.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/mathematical_reasoning/draft_sd/k4_temp0.6_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.6 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/mathematical_reasoning/draft_sd/k4_temp0.3_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.3 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/long_context_completion/draft_sd/k4_temp1.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/long_context_completion/draft_sd/k4_temp0.3_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.3 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/E_temperature_sweep/long_context_completion/draft_sd/k4_temp0.6_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_context_completion" \
  --run_name "long_context_completion" \
  --experiment "E_temperature_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 2048 \
  --temperature 0.6 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/code_gen/draft_sd/k8_temp0.0_draft_Qwen3_4B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-4B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/code_gen/draft_sd/k4_temp0.0_draft_Qwen3_4B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "code_gen" \
  --run_name "code_gen" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-4B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/mathematical_reasoning/draft_sd/k4_temp0.0_draft_Qwen3_1.7B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-1.7B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/mathematical_reasoning/draft_sd/k4_temp0.0_draft_Qwen3_4B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-4B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/mathematical_reasoning/draft_sd/k8_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/mathematical_reasoning/draft_sd/k4_temp0.0_draft_Qwen3_0.6B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-0.6B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/mathematical_reasoning/draft_sd/k8_temp0.0_draft_Qwen3_1.7B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-1.7B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/mathematical_reasoning/draft_sd/k8_temp0.0_draft_Qwen3_4B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "mathematical_reasoning" \
  --run_name "mathematical_reasoning" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-4B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/long_chain_reasoning/draft_sd/k8_temp0.0_draft_Qwen3_4B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-4B" \
  --method "draft_sd" \
  --k 8 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75

# /fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2/D_draft_model_sweep/long_chain_reasoning/draft_sd/k4_temp0.0_draft_Qwen3_4B
python src/phase2_workload_characterization/run_benchmark.py \
  --workload "long_chain_reasoning" \
  --run_name "long_chain_reasoning" \
  --experiment "D_draft_model_sweep" \
  --input_jsonl "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning.jsonl" \
  --limit 100 \
  --max_prompt_tokens 0 \
  --num_turns 1 \
  --out_dir "/fsx/jmanvi/Internship_project/ASD/results/Runs/Phase2" \
  --model "Qwen/Qwen3-8B" \
  --draft_model "Qwen/Qwen3-4B" \
  --method "draft_sd" \
  --k 4 \
  --ngram_lookup_min 1 \
  --ngram_lookup_max 4 \
  --max_tokens 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_logprobs 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --seed 42 \
  --enforce_eager        # disable CUDA graphs (avoids illegal-memory-access) \
  --gpu_memory_utilization 0.75
