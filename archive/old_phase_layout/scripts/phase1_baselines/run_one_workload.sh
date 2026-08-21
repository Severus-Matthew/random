#!/usr/bin/env bash
set -euo pipefail

WORKLOAD="${1:?workload required}"
METHOD="${2:-${DEFAULT_METHOD:-eagle3}}"
K="${3:-4}"

LIMIT="${LIMIT:-50}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SEED="${SEED:-42}"
INPUT_JSONL="${INPUT_JSONL:-}"
RUN_NAME="${RUN_NAME:-$WORKLOAD}"

mkdir -p results logs

if [[ -z "$INPUT_JSONL" ]]; then
  echo "[ERROR] INPUT_JSONL is not set. This patched runner expects the exact benchmark jsonl path."
  exit 2
fi

if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "[ERROR] INPUT_JSONL does not exist: $INPUT_JSONL"
  exit 2
fi

if [[ "$METHOD" == "eagle3" && -z "${EAGLE3_MODEL:-}" ]]; then
  echo "[ERROR] METHOD=eagle3 but EAGLE3_MODEL is empty."
  echo "Set it before launching, e.g.: export EAGLE3_MODEL=/path/or/hf-id/of/eagle3-draft"
  exit 2
fi

echo "[RUN_ONE] gpu=${CUDA_VISIBLE_DEVICES:-ALL} run_name=$RUN_NAME workload=$WORKLOAD method=$METHOD k=$K input=$INPUT_JSONL max_tokens=$MAX_TOKENS limit=$LIMIT"

python /fsx/jmanvi/Internship_project/ASD/src/phase1_baselines/run_benchmark.py \
  --workload "$WORKLOAD" \
  --run_name "$RUN_NAME" \
  --method "$METHOD" \
  --k "$K" \
  --input_jsonl "$INPUT_JSONL" \
  --limit "$LIMIT" \
  --out_dir results \
  --model "${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
  --draft_model "${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --eagle3_model "${EAGLE3_MODEL:-}" \
  --max_tokens "$MAX_TOKENS" \
  --seed "$SEED" \
  --tensor_parallel_size "${TP:-1}"
