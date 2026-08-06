#!/usr/bin/env bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MANIFEST="${1:-results/apexp_block/config_full/manifest.tsv}"
OUT_ROOT="${OUT_ROOT:-$(dirname "$MANIFEST")}"
N_GPUS="${N_GPUS:-8}"

RUNNER=$(grep -Rsl 'Phase 1 + Phase 2 benchmark harness' . --include="run_benchmark.py" | head -1)

if [[ -z "$RUNNER" ]]; then
  echo "ERROR: could not find correct run_benchmark.py"
  exit 1
fi

VLLM_PATH=$(python - <<'PY'
import vllm, inspect, os
print(os.path.dirname(inspect.getfile(vllm)))
PY
)

if ! grep -q "APEXP_BLOCK_TRACE_PATCH" "$VLLM_PATH/v1/core/sched/scheduler.py"; then
  echo "ERROR: vLLM scheduler block trace patch not found."
  echo "Run: python scripts/apexp/patch_vllm_scheduler_block_logger.py"
  exit 1
fi

mkdir -p "$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT/status"

STATUS_FILE="$OUT_ROOT/job_status.tsv"
echo -e "job_id\tgpu\tstatus\tpriority\tstart_time\tend_time\trun_dir" > "$STATUS_FILE"

echo "RUNNER=$RUNNER"
echo "MANIFEST=$MANIFEST"
echo "OUT_ROOT=$OUT_ROOT"
echo "N_GPUS=$N_GPUS"

clean_na() {
  local x="$1"
  if [[ "$x" == "NA" ]]; then
    echo ""
  else
    echo "$x"
  fi
}

run_worker() {
  local GPU="$1"
  local WORKER_LOG="$OUT_ROOT/logs/gpu_${GPU}.log"

  {
    echo "===== Worker GPU $GPU started at $(date) ====="

    python - "$MANIFEST" "$GPU" "$N_GPUS" <<'PY' |
import csv
import sys

manifest = sys.argv[1]
gpu = int(sys.argv[2])
n_gpus = int(sys.argv[3])

with open(manifest, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for i, row in enumerate(reader):
        if i % n_gpus == gpu:
            cols = [
                "priority",
                "job_id",
                "source_experiments",
                "phase",
                "workload",
                "input_jsonl",
                "method",
                "k",
                "temperature",
                "ngram_lookup_min",
                "ngram_lookup_max",
                "draft_model",
                "eagle3_model",
                "target_model",
                "max_prompt_tokens",
                "num_turns",
                "limit",
                "max_tokens",
                "seed",
                "run_dir",
            ]
            print("\t".join(row.get(c, "NA") for c in cols))
PY
    while IFS=$'\t' read -r PRIORITY JOB_ID SOURCE_EXPERIMENTS PHASE WORKLOAD INPUT_JSONL METHOD K TEMPERATURE NGRAM_MIN NGRAM_MAX DRAFT_MODEL EAGLE3_MODEL TARGET_MODEL MAX_PROMPT_TOKENS NUM_TURNS LIMIT MAX_TOKENS SEED RUN_DIR; do

      NGRAM_MIN=$(clean_na "$NGRAM_MIN")
      NGRAM_MAX=$(clean_na "$NGRAM_MAX")
      DRAFT_MODEL=$(clean_na "$DRAFT_MODEL")
      EAGLE3_MODEL=$(clean_na "$EAGLE3_MODEL")
      TARGET_MODEL=$(clean_na "$TARGET_MODEL")
      MAX_PROMPT_TOKENS=$(clean_na "$MAX_PROMPT_TOKENS")
      NUM_TURNS=$(clean_na "$NUM_TURNS")

      echo
      echo "================================================================"
      echo "GPU=$GPU PRIORITY=$PRIORITY JOB=$JOB_ID"
      echo "EXPERIMENTS=$SOURCE_EXPERIMENTS"
      echo "WORKLOAD=$WORKLOAD METHOD=$METHOD K=$K TEMP=$TEMPERATURE"
      echo "INPUT=$INPUT_JSONL"
      echo "RUN_DIR=$RUN_DIR"
      echo "START=$(date)"
      echo "================================================================"

      if [[ -z "$RUN_DIR" || "$RUN_DIR" == "NA" ]]; then
        echo "BAD RUN_DIR for job $JOB_ID"
        echo -e "${JOB_ID}\t${GPU}\tBAD_RUN_DIR\t${PRIORITY}\t$(date)\t$(date)\t${RUN_DIR}" >> "$STATUS_FILE"
        continue
      fi

      if [[ ! -f "$INPUT_JSONL" ]]; then
        echo "MISSING INPUT_JSONL: $INPUT_JSONL"
        echo -e "${JOB_ID}\t${GPU}\tMISSING_INPUT\t${PRIORITY}\t$(date)\t$(date)\t${RUN_DIR}" >> "$STATUS_FILE"
        continue
      fi

      mkdir -p "$RUN_DIR"

      if [[ "${FORCE:-0}" != "1" ]]; then
        if find "$RUN_DIR" -name "summary.csv" | grep -q .; then
          if [[ "$METHOD" == "ar" ]]; then
            echo "SKIP completed AR job: $JOB_ID"
            echo -e "${JOB_ID}\t${GPU}\tSKIPPED\t${PRIORITY}\t$(date)\t$(date)\t${RUN_DIR}" >> "$STATUS_FILE"
            continue
          fi
          if [[ -s "$RUN_DIR/block_events.jsonl" ]]; then
            echo "SKIP completed spec job: $JOB_ID"
            echo -e "${JOB_ID}\t${GPU}\tSKIPPED\t${PRIORITY}\t$(date)\t$(date)\t${RUN_DIR}" >> "$STATUS_FILE"
            continue
          fi
        fi
      fi

      EXTRA_ARGS=()

      if [[ "$METHOD" == "ngram_sd" ]]; then
        if [[ -n "$NGRAM_MIN" ]]; then
          EXTRA_ARGS+=(--ngram_lookup_min "$NGRAM_MIN")
        fi
        if [[ -n "$NGRAM_MAX" ]]; then
          EXTRA_ARGS+=(--ngram_lookup_max "$NGRAM_MAX")
        fi
      fi

      if [[ "$METHOD" == "draft_sd" ]]; then
        if [[ -z "$DRAFT_MODEL" ]]; then
          echo "ERROR: draft_sd missing draft_model"
          echo -e "${JOB_ID}\t${GPU}\tMISSING_DRAFT_MODEL\t${PRIORITY}\t$(date)\t$(date)\t${RUN_DIR}" >> "$STATUS_FILE"
          continue
        fi
        EXTRA_ARGS+=(--draft_model "$DRAFT_MODEL")
        GPU_MEM="0.75"
      else
        GPU_MEM="0.88"
      fi

      if [[ "$METHOD" == "eagle3" ]]; then
        if [[ -n "$EAGLE3_MODEL" ]]; then
          EXTRA_ARGS+=(--eagle3_model "$EAGLE3_MODEL")
        fi
      fi

      if [[ -n "$TARGET_MODEL" ]]; then
        EXTRA_ARGS+=(--model "$TARGET_MODEL")
      fi

      if [[ -n "$MAX_PROMPT_TOKENS" && "$MAX_PROMPT_TOKENS" != "0" ]]; then
        EXTRA_ARGS+=(--max_prompt_tokens "$MAX_PROMPT_TOKENS")
      fi

      if [[ -n "$NUM_TURNS" && "$NUM_TURNS" != "1" ]]; then
        EXTRA_ARGS+=(--num_turns "$NUM_TURNS")
      fi

      START_TIME=$(date)
      set +e
      if [[ "$METHOD" == "ar" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" \
        APEXP_BLOCK_TRACE=0 \
        APEXP_TRACE_STRICT=0 \
        APEXP_TRACE_DIR="$RUN_DIR" \
        APEXP_RUN_ID="$JOB_ID" \
        APEXP_WORKLOAD="$WORKLOAD" \
        APEXP_METHOD="$METHOD" \
        APEXP_K="$K" \
        APEXP_TEMPERATURE="$TEMPERATURE" \
        APEXP_SEED="$SEED" \
        python "$RUNNER" \
          --workload "$WORKLOAD" \
          --input_jsonl "$INPUT_JSONL" \
          --out_dir "$RUN_DIR" \
          --experiment "$SOURCE_EXPERIMENTS" \
          --run_name "${WORKLOAD}_full" \
          --method "$METHOD" \
          --k "$K" \
          --limit "$LIMIT" \
          --temperature "$TEMPERATURE" \
          --max_tokens "$MAX_TOKENS" \
          --seed "$SEED" \
          --enforce_eager \
          --gpu_memory_utilization "$GPU_MEM" \
          "${EXTRA_ARGS[@]}"
        RC=$?
      else
        CUDA_VISIBLE_DEVICES="$GPU" \
        APEXP_BLOCK_TRACE=1 \
        APEXP_TRACE_STRICT=0 \
        APEXP_TRACE_DIR="$RUN_DIR" \
        APEXP_RUN_ID="$JOB_ID" \
        APEXP_WORKLOAD="$WORKLOAD" \
        APEXP_METHOD="$METHOD" \
        APEXP_K="$K" \
        APEXP_TEMPERATURE="$TEMPERATURE" \
        APEXP_SEED="$SEED" \
        python "$RUNNER" \
          --workload "$WORKLOAD" \
          --input_jsonl "$INPUT_JSONL" \
          --out_dir "$RUN_DIR" \
          --experiment "$SOURCE_EXPERIMENTS" \
          --run_name "${WORKLOAD}_full" \
          --method "$METHOD" \
          --k "$K" \
          --limit "$LIMIT" \
          --temperature "$TEMPERATURE" \
          --max_tokens "$MAX_TOKENS" \
          --seed "$SEED" \
          --enforce_eager \
          --gpu_memory_utilization "$GPU_MEM" \
          "${EXTRA_ARGS[@]}"
        RC=$?

        if [[ "$RC" == "0" ]]; then
          python src/apexp/trace/validate_block_events.py "$RUN_DIR/block_events.jsonl"
          RC=$?
        fi
      fi

      set -e
      END_TIME=$(date)

      if [[ "$RC" == "0" ]]; then
        echo "SUCCESS $JOB_ID"
        echo -e "${JOB_ID}\t${GPU}\tSUCCESS\t${PRIORITY}\t${START_TIME}\t${END_TIME}\t${RUN_DIR}" >> "$STATUS_FILE"
      else
        echo "FAILED $JOB_ID RC=$RC"
        echo -e "${JOB_ID}\t${GPU}\tFAILED_${RC}\t${PRIORITY}\t${START_TIME}\t${END_TIME}\t${RUN_DIR}" >> "$STATUS_FILE"
      fi

      echo "END=$(date)"
    done

    echo "===== Worker GPU $GPU finished at $(date) ====="
  } > "$WORKER_LOG" 2>&1
}

for GPU in $(seq 0 $((N_GPUS - 1))); do
  run_worker "$GPU" &
done

wait

echo
echo "All workers done."
echo "Status:"
cat "$STATUS_FILE"
