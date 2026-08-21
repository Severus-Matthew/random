#!/usr/bin/env bash
set -euo pipefail

# Use this INSIDE your existing srun allocation with 8 visible H200s.
# Each dataset gets one GPU via CUDA_VISIBLE_DEVICES, so you get max parallelism.
# Do not submit this with sbatch from inside the allocation; run: bash launch_8gpu_inside_srun.sh

mkdir -p logs results

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export EAGLE3_MODEL="${EAGLE3_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
export TP="${TP:-1}"
export LIMIT="${LIMIT:-50}"
export SEED="${SEED:-42}"

if [[ -z "${EAGLE3_MODEL:-}" ]]; then
  echo "[ERROR] EAGLE3_MODEL is empty."
  echo "Set it first, for example:"
  echo "  export EAGLE3_MODEL=/fsx/jmanvi/path/to/eagle3/model"
  exit 2
fi

TASKS=(
  code_gen
  conversational_generation_gen
  conversational_generation_sft
  hardware_gen
  long_chain_reasoning
  long_context_completion
  long_horizon_swe
  mathematical_reasoning
)

echo "[INFO] Launching ${#TASKS[@]} dataset jobs on GPUs 0-$(( ${#TASKS[@]} - 1 ))"
echo "[INFO] TARGET_MODEL=$TARGET_MODEL"
echo "[INFO] DRAFT_MODEL=$DRAFT_MODEL"
echo "[INFO] EAGLE3_MODEL=$EAGLE3_MODEL"
echo "[INFO] LIMIT=$LIMIT TP=$TP SEED=$SEED"

pids=()
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  gpu="$i"
  log="logs/${task}_gpu${gpu}.log"
  echo "[LAUNCH] $task on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" bash "/fsx/jmanvi/Internship_project/ASD/scripts/phase1_baselines/run_${task}.sh" > "$log" 2>&1 &
  pids+=("$!")
done

echo "[INFO] Started PIDs: ${pids[*]}"
echo "[INFO] Monitor with: tail -f logs/<task>_gpu<id>.log"
echo "[INFO] GPU monitor: watch -n 2 nvidia-smi"

fail=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  task="${TASKS[$idx]}"
  if wait "$pid"; then
    echo "[DONE] $task"
  else
    echo "[FAIL] $task"
    fail=1
  fi
done

exit "$fail"
