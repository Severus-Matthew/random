#!/bin/bash
set -euo pipefail
JOB_LIST="slurm_logs/paper_eval_cap50/baselines/submit_all_jobs.txt" \
MAX_ACTIVE="${MAX_ACTIVE:-6}" \
POLL_SEC="${POLL_SEC:-60}" \
SUBMIT_GAP_SEC="${SUBMIT_GAP_SEC:-30}" \
bash slurm_logs/paper_eval_cap50/submit_throttled.sh
