#!/usr/bin/env bash
set -euo pipefail

JOBS_TSV="${1:?Usage: bash scripts/apexp/run_baseline_matrix_8gpu.sh /path/jobs.tsv}"

cd /fsx/jmanvi/Internship_project/ASD

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

export BENCHMARK_PY="/fsx/jmanvi/Internship_project/ASD/src/layer0_data_collection/run_benchmark.py"
if [[ ! -f "$BENCHMARK_PY" ]]; then
  echo "[baseline] ERROR: BENCHMARK_PY not found: $BENCHMARK_PY"
  exit 2
fi

export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
export EAGLE3_MODEL="${EAGLE3_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"

N_GPUS="${N_GPUS:-8}"
FORCE="${FORCE:-0}"

ROOT_DIR="$(dirname "$JOBS_TSV")"
STATUS_FILE="$ROOT_DIR/baseline_status.tsv"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[baseline] benchmark entrypoint: $BENCHMARK_PY"
echo "[baseline] model: $MODEL"
echo "[baseline] draft_model: $DRAFT_MODEL"
echo "[baseline] eagle3_model: $EAGLE3_MODEL"

echo -e "job_id\tgpu\tstatus\tmethod\tk\tworkload\tstart_time\tend_time\trun_dir" > "$STATUS_FILE"

mapfile -t JOB_LINES < <(tail -n +2 "$JOBS_TSV")
N_JOBS="${#JOB_LINES[@]}"

echo "[baseline] jobs: $N_JOBS"
echo "[baseline] GPUs: $N_GPUS"
echo "[baseline] root: $ROOT_DIR"

run_one_job() {
  local gpu="$1"
  local line="$2"

  IFS=$'\t' read -r JOB_ID PRIORITY WORKLOAD INPUT_FILE METHOD K TEMPERATURE LIMIT MAX_TOKENS RUN_DIR <<< "$line"

  if [[ -z "${RUN_DIR:-}" || "$RUN_DIR" == "NA" ]]; then
    echo "[gpu $gpu] BAD_RUN_DIR job=$JOB_ID line=$line"
    echo -e "${JOB_ID}\t${gpu}\tBAD_RUN_DIR\t${METHOD}\t${K}\t${WORKLOAD}\t$(date)\t$(date)\t${RUN_DIR:-}" >> "$STATUS_FILE"
    return 0
  fi

  mkdir -p "$RUN_DIR"

  local SUMMARY="$RUN_DIR/summary.csv"
  local TRACE="$RUN_DIR/traces.jsonl"

  if [[ "$FORCE" != "1" && -s "$SUMMARY" && -s "$TRACE" ]]; then
    echo "[gpu $gpu] SKIP $JOB_ID $WORKLOAD $METHOD k=$K"
    echo -e "${JOB_ID}\t${gpu}\tSKIP\t${METHOD}\t${K}\t${WORKLOAD}\t$(date)\t$(date)\t${RUN_DIR}" >> "$STATUS_FILE"
    return 0
  fi

  echo "[gpu $gpu] START $JOB_ID $WORKLOAD $METHOD k=$K"

  export CUDA_VISIBLE_DEVICES="$gpu"

  # Baselines should not use online fast control.
  unset APEXP_ONLINE_FAST
  unset APEXP_REQUEST_CONTROL_FILE
  unset APEXP_ONLINE_FAST_TRACE
  unset APEXP_FORCE_ACTIVE_K
  unset APEXP_INITIAL_K
  unset APEXP_DEFAULT_K

  export APEXP_WORKLOAD="$WORKLOAD"
  export APEXP_METHOD="$METHOD"
  export APEXP_K="$K"
  export APEXP_TEMPERATURE="$TEMPERATURE"
  export APEXP_SEED="${SEED:-42}"

  if [[ "$METHOD" == "ar" ]]; then
    export APEXP_BLOCK_TRACE=0
    export APEXP_TRACE_STRICT=0
  else
    export APEXP_BLOCK_TRACE=1
    export APEXP_TRACE_STRICT=0
    export APEXP_TRACE_DIR="$RUN_DIR/trace"
    export APEXP_RUN_ID="$JOB_ID"
    mkdir -p "$APEXP_TRACE_DIR"
  fi

  local COMMON_ARGS=(
    --workload "$WORKLOAD"
    --run_name "$JOB_ID"
    --experiment "baseline_matrix_all_methods_all_k"
    --input_jsonl "$INPUT_FILE"
    --limit "$LIMIT"
    --out_dir "$RUN_DIR"
    --model "$MODEL"
    --method "$METHOD"
    --max_tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p 1.0
    --tensor_parallel_size 1
    --seed "${SEED:-42}"
  )

  local EXTRA_ARGS=()

  if [[ "$METHOD" == "ngram_sd" ]]; then
    EXTRA_ARGS+=(--k "$K")
    EXTRA_ARGS+=(--ngram_lookup_min 1 --ngram_lookup_max "$K")
  elif [[ "$METHOD" == "draft_sd" ]]; then
    EXTRA_ARGS+=(--k "$K")
    EXTRA_ARGS+=(--draft_model "$DRAFT_MODEL")
    EXTRA_ARGS+=(--gpu_memory_utilization 0.80)
    EXTRA_ARGS+=(--enforce_eager)
  elif [[ "$METHOD" == "eagle3" ]]; then
    EXTRA_ARGS+=(--k "$K")
    EXTRA_ARGS+=(--eagle3_model "$EAGLE3_MODEL")
  fi

  local START
  START="$(date)"

  set +e
  python3 "$BENCHMARK_PY" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/${JOB_ID}_${WORKLOAD}_${METHOD}_k${K}_gpu${gpu}.log" 2>&1
  RC=$?
  set -e

  local END
  END="$(date)"

  if [[ "$RC" == "0" ]]; then
    echo "[gpu $gpu] DONE $JOB_ID $WORKLOAD $METHOD k=$K"
    echo -e "${JOB_ID}\t${gpu}\tSUCCESS\t${METHOD}\t${K}\t${WORKLOAD}\t${START}\t${END}\t${RUN_DIR}" >> "$STATUS_FILE"
  else
    echo "[gpu $gpu] FAIL $JOB_ID $WORKLOAD $METHOD k=$K rc=$RC"
    echo -e "${JOB_ID}\t${gpu}\tFAILED_${RC}\t${METHOD}\t${K}\t${WORKLOAD}\t${START}\t${END}\t${RUN_DIR}" >> "$STATUS_FILE"
  fi
}

worker() {
  local gpu="$1"

  for ((idx=gpu; idx<N_JOBS; idx+=N_GPUS)); do
    run_one_job "$gpu" "${JOB_LINES[$idx]}"
  done
}

pids=()
for ((gpu=0; gpu<N_GPUS; gpu++)); do
  worker "$gpu" &
  pids+=("$!")
done

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    fail=1
  fi
done

echo "[baseline] all workers complete"
exit "$fail"
