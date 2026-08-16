#!/bin/bash
set -euo pipefail
JOB_LIST="slurm_logs/paper_eval_cap50_profile_kgrid/apex/submit_all_jobs.txt" \
MAX_ACTIVE=4 \
POLL_SEC=60 \
SUBMIT_GAP_SEC=60 \
bash slurm_logs/paper_eval_cap50_profile_kgrid/submit_throttled.sh
