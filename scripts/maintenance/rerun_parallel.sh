#!/usr/bin/env bash
# Auto-generated: re-run failed jobs across NGPU GPUs, one task per GPU.
# Uses a flock work-queue so a freed GPU immediately pulls the next job
# (load-balanced — better than round-robin when job durations vary).
set -u
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # required: parent inits CUDA via set_seed

NGPU=${NGPU:-8}
CMDS_FILE="/fsx/jmanvi/Internship_project/ASD/src/util/commands.txt"
LOGDIR="${LOGDIR:-rerun_logs}"
mkdir -p "$LOGDIR"

mapfile -t CMDS < "$CMDS_FILE"
echo 0 > .queue.idx
: > .queue.lock

run_worker () {
  local gpu=$1
  while :; do
    exec 9>>.queue.lock; flock 9
    local idx; idx=$(<.queue.idx)
    if [ "$idx" -ge "${#CMDS[@]}" ]; then flock -u 9; break; fi
    echo $((idx + 1)) > .queue.idx
    flock -u 9
    local cmd="${CMDS[$idx]}"
    [ -z "$cmd" ] && continue
    echo "[GPU $gpu] job $idx/${#CMDS[@]}"
    CUDA_VISIBLE_DEVICES=$gpu $cmd > "$LOGDIR/gpu${gpu}_job${idx}.log" 2>&1 \
      || echo "[GPU $gpu] job $idx FAILED (see $LOGDIR/gpu${gpu}_job${idx}.log)"
  done
}

for g in $(seq 0 $((NGPU - 1))); do run_worker "$g" & done
wait
echo "All workers done. Check $LOGDIR/ for any FAILED jobs, then re-aggregate."
