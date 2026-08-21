#!/bin/bash
set -uo pipefail

MAX_ACTIVE="${MAX_ACTIVE:-6}"
POLL_SEC="${POLL_SEC:-60}"
SUBMIT_GAP_SEC="${SUBMIT_GAP_SEC:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_LIST="${JOB_LIST:-$SCRIPT_DIR/submit_all_jobs.txt}"

if [ ! -f "$JOB_LIST" ]; then
  echo "[error] job list not found: $JOB_LIST"
  exit 1
fi

mapfile -t JOB_SCRIPTS < <(grep -v '^[[:space:]]*$' "$JOB_LIST" | grep -v '^[[:space:]]*#')

TOTAL="${#JOB_SCRIPTS[@]}"
ACTIVE_JOBS=()
SUBMITTED=0
FAILED_SUBMIT=0

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

prune_active_jobs() {
  local kept=()
  local jid

  for jid in "${ACTIVE_JOBS[@]}"; do
    if squeue -h -j "$jid" 2>/dev/null | grep -q .; then
      kept+=("$jid")
    else
      echo "[$(timestamp)] finished/disappeared from queue: $jid"
    fi
  done

  ACTIVE_JOBS=("${kept[@]}")
}

submit_job() {
  local script="$1"

  if [ ! -f "$script" ]; then
    echo "[$(timestamp)] [skip missing] $script"
    FAILED_SUBMIT=$((FAILED_SUBMIT + 1))
    return 0
  fi

  local out
  local jid

  echo "[$(timestamp)] submitting: $script"

  out=$(sbatch "$script" 2>&1)
  local rc=$?

  if [ "$rc" -ne 0 ]; then
    echo "[$(timestamp)] [submit failed] $script"
    echo "$out"
    FAILED_SUBMIT=$((FAILED_SUBMIT + 1))
    return 0
  fi

  echo "$out"
  jid=$(echo "$out" | awk '/Submitted batch job/ {print $4}' | tail -1)

  if [ -z "$jid" ]; then
    echo "[$(timestamp)] [warning] could not parse job id for: $script"
    FAILED_SUBMIT=$((FAILED_SUBMIT + 1))
    return 0
  fi

  ACTIVE_JOBS+=("$jid")
  SUBMITTED=$((SUBMITTED + 1))

  echo "[$(timestamp)] active job ids: ${ACTIVE_JOBS[*]}"
  echo "[$(timestamp)] submitted $SUBMITTED / $TOTAL"
}

echo "[$(timestamp)] starting throttled submission"
echo "[$(timestamp)] job list: $JOB_LIST"
echo "[$(timestamp)] total jobs: $TOTAL"
echo "[$(timestamp)] max active jobs: $MAX_ACTIVE"
echo

idx=0

while [ "$idx" -lt "$TOTAL" ] || [ "${#ACTIVE_JOBS[@]}" -gt 0 ]; do
  prune_active_jobs

  while [ "$idx" -lt "$TOTAL" ] && [ "${#ACTIVE_JOBS[@]}" -lt "$MAX_ACTIVE" ]; do
    submit_job "${JOB_SCRIPTS[$idx]}"
    idx=$((idx + 1))

    sleep "$SUBMIT_GAP_SEC"
    prune_active_jobs
  done

  echo "[$(timestamp)] status: active=${#ACTIVE_JOBS[@]} submitted=$SUBMITTED/$TOTAL remaining=$((TOTAL - idx)) failed_submit=$FAILED_SUBMIT"

  if [ "$idx" -lt "$TOTAL" ] || [ "${#ACTIVE_JOBS[@]}" -gt 0 ]; then
    sleep "$POLL_SEC"
  fi
done

echo
echo "[$(timestamp)] all submitted jobs have finished or left queue"
echo "[$(timestamp)] total submitted: $SUBMITTED"
echo "[$(timestamp)] submit failures/skips: $FAILED_SUBMIT"
