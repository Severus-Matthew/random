#!/bin/bash
set -uo pipefail

JOB_LIST="${JOB_LIST:-slurm_logs/paper_eval_final_full6/submit_all_jobs.txt}"
MAX_ACTIVE="${MAX_ACTIVE:-10}"
POLL_SEC="${POLL_SEC:-60}"
SUBMIT_GAP_SEC="${SUBMIT_GAP_SEC:-10}"

mapfile -t JOB_SCRIPTS < <(grep -v '^[[:space:]]*$' "$JOB_LIST" | grep -v '^[[:space:]]*#')

ACTIVE_JOBS=()
idx=0
TOTAL="${#JOB_SCRIPTS[@]}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

prune_active_jobs() {
  local kept=()
  for jid in "${ACTIVE_JOBS[@]}"; do
    if squeue -h -j "$jid" 2>/dev/null | grep -q .; then
      kept+=("$jid")
    else
      echo "[$(timestamp)] finished/disappeared: $jid"
    fi
  done
  ACTIVE_JOBS=("${kept[@]}")
}

echo "[$(timestamp)] JOB_LIST=$JOB_LIST"
echo "[$(timestamp)] TOTAL=$TOTAL MAX_ACTIVE=$MAX_ACTIVE"

while [ "$idx" -lt "$TOTAL" ] || [ "${#ACTIVE_JOBS[@]}" -gt 0 ]; do
  prune_active_jobs

  while [ "$idx" -lt "$TOTAL" ] && [ "${#ACTIVE_JOBS[@]}" -lt "$MAX_ACTIVE" ]; do
    script="${JOB_SCRIPTS[$idx]}"
    echo "[$(timestamp)] submitting $script"
    out=$(sbatch "$script" 2>&1)
    echo "$out"

    jid=$(echo "$out" | awk '/Submitted batch job/ {print $4}' | tail -1)
    if [ -n "$jid" ]; then
      ACTIVE_JOBS+=("$jid")
    else
      echo "[$(timestamp)] WARN: could not parse job id for $script"
    fi

    idx=$((idx + 1))
    sleep "$SUBMIT_GAP_SEC"
    prune_active_jobs
  done

  echo "[$(timestamp)] active=${#ACTIVE_JOBS[@]} submitted_index=$idx/$TOTAL"
  if [ "$idx" -lt "$TOTAL" ] || [ "${#ACTIVE_JOBS[@]}" -gt 0 ]; then
    sleep "$POLL_SEC"
  fi
done

echo "[$(timestamp)] all done"
