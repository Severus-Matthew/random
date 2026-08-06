# #!/usr/bin/env bash
# # =============================================================================
# # run_full_study.sh
# # Reads experiments_config.json, generates all jobs, runs them across
# # 8 H200 GPUs using a proper queue (max 8 concurrent).
# #
# # Usage:
# #   bash run_full_study.sh                              # run everything
# #   bash run_full_study.sh --dry-run                    # print jobs, don't run
# #   bash run_full_study.sh --experiment A               # only experiment A_*
# #   bash run_full_study.sh --skip-existing              # skip jobs that already
# #                                                       # have a summary.csv
# #   bash run_full_study.sh --jobs-file /tmp/jobs.jsonl  # bypass gen_jobs.py
# #   bash run_full_study.sh --config configs/retry.json  # use different config
# #
# # --skip-existing logic:
# #   Generates ALL jobs first, then uses Python to filter out any job whose
# #   exact output directory already contains a summary.csv.
# #   Path reconstruction matches run_benchmark.py exactly so no job is
# #   incorrectly skipped or missed.
# # =============================================================================
# set -euo pipefail

# # ── Paths ─────────────────────────────────────────────────────────────────
# VENV="/fsx/jmanvi/Internship_project/.venv"
# ASD="/fsx/jmanvi/Internship_project/ASD"
# BENCHMARK="${ASD}/src/phase2_workload_characterization/run_benchmark.py"
# GEN_JOBS="${ASD}/src/phase2_workload_characterization/gen_jobs.py"
# EXP_CONFIG="${ASD}/configs/experiments_config.json"
# LOG_DIR="${ASD}/logs/full_study"
# TMP_DIR="/fsx/jmanvi/Internship_project/tmp"
# NUM_GPUS=8

# # ── Flags ─────────────────────────────────────────────────────────────────
# DRY_RUN=0
# FILTER_EXP=""
# SKIP_EXISTING=0
# JOBS_FILE_OVERRIDE=""
# CONFIG_OVERRIDE=""

# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --dry-run)       DRY_RUN=1 ;;
#         --experiment)    FILTER_EXP="$2";         shift ;;
#         --skip-existing) SKIP_EXISTING=1 ;;
#         --jobs-file)     JOBS_FILE_OVERRIDE="$2"; shift ;;
#         --config)        CONFIG_OVERRIDE="$2";    shift ;;
#         *) echo "Unknown arg: $1"; exit 1 ;;
#     esac
#     shift
# done

# [[ -n "${CONFIG_OVERRIDE}" ]] && EXP_CONFIG="${CONFIG_OVERRIDE}"

# mkdir -p "${LOG_DIR}" "${TMP_DIR}"
# source "${VENV}/bin/activate"

# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export VLLM_ENGINE_MULTIPROC_METHOD=spawn
# export TOKENIZERS_PARALLELISM=false

# # ── Clear stale torch compile cache ───────────────────────────────────────
# COMPILE_CACHE="${HOME}/.cache/vllm/torch_compile_cache"
# if [[ -d "${COMPILE_CACHE}" ]]; then
#     echo "[study] Clearing stale torch compile cache"
#     rm -rf "${COMPILE_CACHE}"
# fi

# # ── Write JSON job parser to a temp file once ─────────────────────────────
# # Using a file avoids the heredoc-steals-stdin bug that occurs when
# # combining  eval "$(echo ... | python - << 'HEREDOC')"
# JOB_PARSER=$(mktemp "${TMP_DIR}/job_parser_XXXXXX.py")
# cat > "${JOB_PARSER}" << 'PYEOF'
# import sys, json, shlex
# j = json.loads(sys.stdin.read())
# for key, val in {
#     "exp":    j["experiment"],
#     "workload":j["workload"],
#     "method": j["method"],
#     "k":      str(j["k"]),
#     "temp":   str(j["temperature"]),
#     "ctx":    str(j["max_prompt_tokens"]),
#     "turns":  str(j["num_turns"]),
#     "nlmin":  str(j["ngram_lookup_min"]),
#     "nlmax":  str(j["ngram_lookup_max"]),
#     "draft":  j["draft_model"],
#     "eagle":  j["eagle3_model"],
#     "jsonl":  j["input_jsonl"],
#     "maxtok": str(j["max_tokens"]),
#     "limit":  str(j["limit"]),
#     "model":  j["model"],
#     "outdir": j["out_dir"],
# }.items():
#     print(f"local {key}={shlex.quote(val)}")
# PYEOF

# # ── Job generation ─────────────────────────────────────────────────────────
# if [[ -n "${JOBS_FILE_OVERRIDE}" ]]; then
#     [[ -f "${JOBS_FILE_OVERRIDE}" ]] || {
#         echo "[ERROR] Jobs file not found: ${JOBS_FILE_OVERRIDE}"; exit 1
#     }
#     JOBS_FILE="${JOBS_FILE_OVERRIDE}"
#     OWNS_JOBS_FILE=0
#     echo "[study] Using pre-generated jobs file: ${JOBS_FILE}"
# else
#     [[ -f "${EXP_CONFIG}" ]] || {
#         echo "[ERROR] Config not found: ${EXP_CONFIG}"; exit 1
#     }
#     JOBS_FILE=$(mktemp "${TMP_DIR}/study_jobs_XXXXXX.jsonl")
#     OWNS_JOBS_FILE=1
#     python "${GEN_JOBS}" \
#         --config  "${EXP_CONFIG}" \
#         --out     "${JOBS_FILE}" \
#         --filter  "${FILTER_EXP}"
# fi

# ALL_JOBS=$(wc -l < "${JOBS_FILE}")
# echo "[study] ${ALL_JOBS} total jobs generated"

# # ── --skip-existing: filter out jobs that already have summary.csv ─────────
# # Uses Python to reconstruct the exact output path (matching run_benchmark.py)
# # and checks for summary.csv. Far more reliable than bash path reconstruction.
# if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
#     FILTERED_FILE=$(mktemp "${TMP_DIR}/study_jobs_filtered_XXXXXX.jsonl")

#     python - "${JOBS_FILE}" "${FILTERED_FILE}" << 'PY'
# import sys, json
# from pathlib import Path

# jobs_path    = sys.argv[1]
# filtered_path= sys.argv[2]

# def expected_summary(j: dict) -> Path:
#     """
#     Reconstruct the exact output path run_benchmark.py uses, then return
#     the path to summary.csv inside it.  Must stay in sync with run_benchmark.py.
#     """
#     method   = j["method"]
#     k        = j["k"]
#     temp     = j["temperature"]
#     ctx      = j["max_prompt_tokens"]
#     turns    = j["num_turns"]
#     nlmin    = j["ngram_lookup_min"]
#     nlmax    = j["ngram_lookup_max"]
#     draft    = j["draft_model"]
#     eagle    = j["eagle3_model"]
#     out_dir  = j["out_dir"]
#     exp      = j["experiment"]
#     run_name = j["run_name"]

#     ngram_tag = f"_lk{nlmin}-{nlmax}" if method == "ngram_sd" else ""
#     temp_tag  = f"_temp{temp}"
#     ctx_tag   = f"_ctx{ctx}"  if ctx   > 0  else ""
#     turn_tag  = f"_t{turns}"  if turns > 1  else ""

#     model_tag = ""
#     if method == "draft_sd" and draft:
#         slug = draft.split("/")[-1].replace("-", "_")
#         model_tag = f"_draft_{slug}"
#     elif method == "eagle3" and eagle:
#         slug = eagle.split("/")[-1].replace("-", "_")
#         model_tag = f"_eagle_{slug}"

#     folder = f"k{k}{ngram_tag}{temp_tag}{ctx_tag}{turn_tag}{model_tag}"
#     return Path(out_dir) / exp / run_name / method / folder / "summary.csv"

# kept = skipped = 0
# with open(jobs_path) as fin, open(filtered_path, "w") as fout:
#     for line in fin:
#         j = json.loads(line)
#         summary = expected_summary(j)
#         if summary.exists():
#             skipped += 1
#         else:
#             fout.write(line)
#             kept += 1

# print(f"[skip ] {skipped} jobs already have summary.csv — skipping")
# print(f"[run  ] {kept} jobs to run")
# PY

#     # If the filtered file is empty, nothing to do
#     REMAINING=$(wc -l < "${FILTERED_FILE}")
#     if [[ "${REMAINING}" -eq 0 ]]; then
#         echo "[study] All jobs already complete. Nothing to run."
#         [[ "${OWNS_JOBS_FILE}" -eq 1 ]] && rm -f "${JOBS_FILE}"
#         rm -f "${FILTERED_FILE}"
#         exit 0
#     fi

#     # Replace jobs file with filtered version
#     [[ "${OWNS_JOBS_FILE}" -eq 1 ]] && rm -f "${JOBS_FILE}"
#     JOBS_FILE="${FILTERED_FILE}"
#     OWNS_JOBS_FILE=1
# fi

# TOTAL=$(wc -l < "${JOBS_FILE}")
# echo "[study] ${TOTAL} jobs to run | ${NUM_GPUS} GPUs | dry=${DRY_RUN}"

# # ── Dry run ────────────────────────────────────────────────────────────────
# if [[ "${DRY_RUN}" -eq 1 ]]; then
#     echo "[dry-run] Jobs to run:"
#     python - "${JOBS_FILE}" << 'PY'
# import sys, json
# with open(sys.argv[1]) as f:
#     for i, line in enumerate(f, 1):
#         j = json.loads(line)
#         dm = f"  draft={j['draft_model'].split('/')[-1]}" if j['draft_model'] else ""
#         print(f"  {i:4d}  {j['experiment']:32s}  {j['workload']:28s}  "
#               f"{j['method']:10s}  k={j['k']}  T={j['temperature']}  "
#               f"ctx={j['max_prompt_tokens']}  t={j['num_turns']}{dm}")
# PY
#     [[ "${OWNS_JOBS_FILE}" -eq 1 ]] && rm -f "${JOBS_FILE}"
#     exit 0
# fi

# # ── GPU queue ──────────────────────────────────────────────────────────────
# declare -A GPU_PIDS
# declare -A GPU_DESC
# declare -A GPU_OUT

# for g in $(seq 0 $((NUM_GPUS-1))); do
#     GPU_PIDS[$g]=""
#     GPU_DESC[$g]=""
#     GPU_OUT[$g]=""
# done

# total_done=0
# total_failed=0
# job_idx=0

# wait_for_free_gpu() {
#     while true; do
#         for gpu in $(seq 0 $((NUM_GPUS-1))); do
#             pid="${GPU_PIDS[$gpu]}"
#             if [[ -z "$pid" ]]; then
#                 echo "$gpu"; return
#             fi
#             if ! kill -0 "$pid" 2>/dev/null; then
#                 wait "$pid" 2>/dev/null || true
#                 local out_key="${GPU_OUT[$gpu]:-}"
#                 if [[ -n "$out_key" ]] && compgen -G "${out_key}/**/summary.csv" > /dev/null 2>&1; then
#                     echo "[done ] GPU=${gpu} ${GPU_DESC[$gpu]}" >&2
#                 else
#                     echo "[FAIL ] GPU=${gpu} ${GPU_DESC[$gpu]} (no summary.csv)" >&2
#                     total_failed=$((total_failed+1))
#                 fi
#                 total_done=$((total_done+1))
#                 echo "[queue] ${total_done}/${TOTAL} done | ${total_failed} failed" >&2
#                 GPU_PIDS[$gpu]=""
#                 GPU_OUT[$gpu]=""
#                 echo "$gpu"; return
#             fi
#         done
#         sleep 3
#     done
# }

# launch_job() {
#     local gpu=$1
#     local jj=$2

#     # Parse all fields — pipe JSON to the pre-written parser file.
#     # This avoids the heredoc-steals-stdin bug where
#     # eval "$(echo ... | python - << 'PY' ... PY)" drops the piped input.
#     eval "$(echo "${jj}" | python "${JOB_PARSER}")"

#     [[ -f "${jsonl}" ]] || { echo "[WARN ] input not found: ${jsonl}" >&2; return; }

#     local gpu_mem_util="0.92"
#     [[ "${method}" == "draft_sd" ]] && gpu_mem_util="0.80"

#     local desc="${exp}/${workload}/${method}/k${k}/T${temp}/ctx${ctx}/t${turns}"
#     GPU_DESC[$gpu]="${desc}"
#     GPU_OUT[$gpu]="${outdir}/${exp}/${workload}/${method}"
#     local log="${LOG_DIR}/${exp}_${workload}_${method}_k${k}_T${temp}_ctx${ctx}_t${turns}.log"

#     CUDA_VISIBLE_DEVICES="${gpu}" \
#     python "${BENCHMARK}" \
#         --experiment            "${exp}" \
#         --workload              "${workload}" \
#         --run_name              "${workload}" \
#         --method                "${method}" \
#         --k                     "${k}" \
#         --temperature           "${temp}" \
#         --ngram_lookup_min      "${nlmin}" \
#         --ngram_lookup_max      "${nlmax}" \
#         --draft_model           "${draft}" \
#         --eagle3_model          "${eagle}" \
#         --input_jsonl           "${jsonl}" \
#         --max_tokens            "${maxtok}" \
#         --max_prompt_tokens     "${ctx}" \
#         --num_turns             "${turns}" \
#         --limit                 "${limit}" \
#         --out_dir               "${outdir}" \
#         --model                 "${model}" \
#         --seed                  42 \
#         --tensor_parallel_size  1 \
#         --gpu_memory_utilization "${gpu_mem_util}" \
#         > "${log}" 2>&1 &

#     GPU_PIDS[$gpu]=$!
# }

# echo "[study] Starting queue..."
# while IFS= read -r job_json; do
#     gpu=$(wait_for_free_gpu)
#     job_idx=$((job_idx+1))
#     desc=$(echo "$job_json" | python -c "
# import sys,json
# j=json.loads(sys.stdin.read())
# dm=(' draft='+j['draft_model'].split('/')[-1]) if j['draft_model'] else ''
# print(f\"{j['experiment']:32s} {j['workload']:28s} {j['method']:10s} k={j['k']} T={j['temperature']} ctx={j['max_prompt_tokens']} t={j['num_turns']}{dm}\")
# ")
#     echo "[launch] ${job_idx}/${TOTAL} GPU=${gpu} ${desc}"
#     launch_job "$gpu" "$job_json"
# done < "${JOBS_FILE}"

# # ── Drain ──────────────────────────────────────────────────────────────────
# echo "[study] Draining final jobs..."
# for gpu in $(seq 0 $((NUM_GPUS-1))); do
#     pid="${GPU_PIDS[$gpu]}"
#     if [[ -n "$pid" ]]; then
#         wait "$pid" 2>/dev/null || true
#         local_out="${GPU_OUT[$gpu]:-}"
#         if [[ -n "$local_out" ]] && compgen -G "${local_out}/**/summary.csv" > /dev/null 2>&1; then
#             echo "[done ] GPU=${gpu} ${GPU_DESC[$gpu]}"
#         else
#             echo "[FAIL ] GPU=${gpu} ${GPU_DESC[$gpu]} (no summary.csv)"
#             total_failed=$((total_failed+1))
#         fi
#         total_done=$((total_done+1))
#     fi
# done

# [[ "${OWNS_JOBS_FILE}" -eq 1 ]] && rm -f "${JOBS_FILE}"
# rm -f "${JOB_PARSER}" 2>/dev/null || true

# echo ""
# echo "=================================================="
# echo "[study] COMPLETE"
# echo "  Total generated : ${ALL_JOBS}"
# echo "  Ran this session: ${TOTAL}"
# echo "  Done            : ${total_done}"
# echo "  Failed          : ${total_failed}"
# echo "  Config          : ${EXP_CONFIG}"
# echo "  Logs            : ${LOG_DIR}"
# echo "=================================================="
# [[ "${total_failed}" -eq 0 ]] || exit 1

#!/usr/bin/env bash
# =============================================================================
# run_full_study.sh
# Reads experiments_config.json, generates all jobs, runs them across
# 8 H200 GPUs using a proper queue (max 8 concurrent).
#
# Usage:
#   bash run_full_study.sh                              # run everything
#   bash run_full_study.sh --dry-run                    # print jobs, don't run
#   bash run_full_study.sh --experiment A               # only experiment A_*
#   bash run_full_study.sh --skip-existing              # skip jobs that already
#                                                       # have a summary.csv
#   bash run_full_study.sh --jobs-file /tmp/jobs.jsonl  # bypass gen_jobs.py
#   bash run_full_study.sh --config configs/retry.json  # use different config
#
# --skip-existing logic:
#   Generates ALL jobs first, then uses Python to filter out any job whose
#   exact output directory already contains a summary.csv.
#   Path reconstruction matches run_benchmark.py exactly so no job is
#   incorrectly skipped or missed.
# =============================================================================
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
VENV="/fsx/jmanvi/Internship_project/.venv"
ASD="/fsx/jmanvi/Internship_project/ASD"
BENCHMARK="${ASD}/src/phase2_workload_characterization/run_benchmark.py"
GEN_JOBS="${ASD}/src/phase2_workload_characterization/gen_jobs.py"
EXP_CONFIG="${ASD}/configs/experiments_config.json"
LOG_DIR="${ASD}/logs/full_study"
TMP_DIR="/fsx/jmanvi/Internship_project/tmp"
NUM_GPUS=8

# ── Flags ─────────────────────────────────────────────────────────────────
DRY_RUN=0
FILTER_EXP=""
SKIP_EXISTING=0
JOBS_FILE_OVERRIDE=""
CONFIG_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1 ;;
        --experiment)    FILTER_EXP="$2";         shift ;;
        --skip-existing) SKIP_EXISTING=1 ;;
        --jobs-file)     JOBS_FILE_OVERRIDE="$2"; shift ;;
        --config)        CONFIG_OVERRIDE="$2";    shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

[[ -n "${CONFIG_OVERRIDE}" ]] && EXP_CONFIG="${CONFIG_OVERRIDE}"

mkdir -p "${LOG_DIR}" "${TMP_DIR}"
source "${VENV}/bin/activate"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# ── Clear stale torch compile cache ───────────────────────────────────────
COMPILE_CACHE="${HOME}/.cache/vllm/torch_compile_cache"
if [[ -d "${COMPILE_CACHE}" ]]; then
    echo "[study] Clearing stale torch compile cache"
    rm -rf "${COMPILE_CACHE}"
fi

# ── Write JSON job parser to a temp file once ─────────────────────────────
# Using a file avoids the heredoc-steals-stdin bug that occurs when
# combining  eval "$(echo ... | python - << 'HEREDOC')"
JOB_PARSER=$(mktemp "${TMP_DIR}/job_parser_XXXXXX.py")
cat > "${JOB_PARSER}" << 'PYEOF'
import sys, json, shlex
j = json.loads(sys.stdin.read())
for key, val in {
    "exp":    j["experiment"],
    "workload":j["workload"],
    "method": j["method"],
    "k":      str(j["k"]),
    "temp":   str(j["temperature"]),
    "ctx":    str(j["max_prompt_tokens"]),
    "turns":  str(j["num_turns"]),
    "nlmin":  str(j["ngram_lookup_min"]),
    "nlmax":  str(j["ngram_lookup_max"]),
    "draft":  j["draft_model"],
    "eagle":  j["eagle3_model"],
    "jsonl":  j["input_jsonl"],
    "maxtok": str(j["max_tokens"]),
    "limit":  str(j["limit"]),
    "model":  j["model"],
    "outdir": j["out_dir"],
}.items():
    print(f"local {key}={shlex.quote(val)}")
PYEOF

# ── Job generation ─────────────────────────────────────────────────────────
if [[ -n "${JOBS_FILE_OVERRIDE}" ]]; then
    [[ -f "${JOBS_FILE_OVERRIDE}" ]] || {
        echo "[ERROR] Jobs file not found: ${JOBS_FILE_OVERRIDE}"; exit 1
    }
    JOBS_FILE="${JOBS_FILE_OVERRIDE}"
    OWNS_JOBS_FILE=0
    echo "[study] Using pre-generated jobs file: ${JOBS_FILE}"
else
    [[ -f "${EXP_CONFIG}" ]] || {
        echo "[ERROR] Config not found: ${EXP_CONFIG}"; exit 1
    }
    JOBS_FILE=$(mktemp "${TMP_DIR}/study_jobs_XXXXXX.jsonl")
    OWNS_JOBS_FILE=1
    python "${GEN_JOBS}" \
        --config  "${EXP_CONFIG}" \
        --out     "${JOBS_FILE}" \
        --filter  "${FILTER_EXP}"
fi

ALL_JOBS=$(wc -l < "${JOBS_FILE}")
echo "[study] ${ALL_JOBS} total jobs generated"

# ── --skip-existing: filter out jobs that already have summary.csv ─────────
# Uses Python to reconstruct the exact output path (matching run_benchmark.py)
# and checks for summary.csv. Far more reliable than bash path reconstruction.
if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
    FILTERED_FILE=$(mktemp "${TMP_DIR}/study_jobs_filtered_XXXXXX.jsonl")

    python - "${JOBS_FILE}" "${FILTERED_FILE}" << 'PY'
import sys, json
from pathlib import Path

jobs_path    = sys.argv[1]
filtered_path= sys.argv[2]

def expected_summary(j: dict) -> Path:
    """
    Reconstruct the exact output path run_benchmark.py uses, then return
    the path to summary.csv inside it.  Must stay in sync with run_benchmark.py.
    """
    method   = j["method"]
    k        = j["k"]
    temp     = j["temperature"]
    ctx      = j["max_prompt_tokens"]
    turns    = j["num_turns"]
    nlmin    = j["ngram_lookup_min"]
    nlmax    = j["ngram_lookup_max"]
    draft    = j["draft_model"]
    eagle    = j["eagle3_model"]
    out_dir  = j["out_dir"]
    exp      = j["experiment"]
    run_name = j["run_name"]

    ngram_tag = f"_lk{nlmin}-{nlmax}" if method == "ngram_sd" else ""
    temp_tag  = f"_temp{temp}"
    ctx_tag   = f"_ctx{ctx}"  if ctx   > 0  else ""
    turn_tag  = f"_t{turns}"  if turns > 1  else ""

    model_tag = ""
    if method == "draft_sd" and draft:
        slug = draft.split("/")[-1].replace("-", "_")
        model_tag = f"_draft_{slug}"
    elif method == "eagle3" and eagle:
        slug = eagle.split("/")[-1].replace("-", "_")
        model_tag = f"_eagle_{slug}"

    folder = f"k{k}{ngram_tag}{temp_tag}{ctx_tag}{turn_tag}{model_tag}"
    return Path(out_dir) / exp / run_name / method / folder / "summary.csv"

kept = skipped = 0
with open(jobs_path) as fin, open(filtered_path, "w") as fout:
    for line in fin:
        j = json.loads(line)
        summary = expected_summary(j)
        if summary.exists():
            skipped += 1
        else:
            fout.write(line)
            kept += 1

print(f"[skip ] {skipped} jobs already have summary.csv — skipping")
print(f"[run  ] {kept} jobs to run")
PY

    # If the filtered file is empty, nothing to do
    REMAINING=$(wc -l < "${FILTERED_FILE}")
    if [[ "${REMAINING}" -eq 0 ]]; then
        echo "[study] All jobs already complete. Nothing to run."
        [[ "${OWNS_JOBS_FILE}" -eq 1 ]] && rm -f "${JOBS_FILE}"
        rm -f "${FILTERED_FILE}"
        exit 0
    fi

    # Replace jobs file with filtered version
    [[ "${OWNS_JOBS_FILE}" -eq 1 ]] && rm -f "${JOBS_FILE}"
    JOBS_FILE="${FILTERED_FILE}"
    OWNS_JOBS_FILE=1
fi

TOTAL=$(wc -l < "${JOBS_FILE}")
echo "[study] ${TOTAL} jobs to run | ${NUM_GPUS} GPUs | dry=${DRY_RUN}"

# ── Dry run ────────────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] Jobs to run:"
    python - "${JOBS_FILE}" << 'PY'
import sys, json
with open(sys.argv[1]) as f:
    for i, line in enumerate(f, 1):
        j = json.loads(line)
        dm = f"  draft={j['draft_model'].split('/')[-1]}" if j['draft_model'] else ""
        print(f"  {i:4d}  {j['experiment']:32s}  {j['workload']:28s}  "
              f"{j['method']:10s}  k={j['k']}  T={j['temperature']}  "
              f"ctx={j['max_prompt_tokens']}  t={j['num_turns']}{dm}")
PY
    [[ "${OWNS_JOBS_FILE}" -eq 1 ]] && rm -f "${JOBS_FILE}"
    exit 0
fi

# ── GPU queue ──────────────────────────────────────────────────────────────
declare -A GPU_PIDS
declare -A GPU_DESC
declare -A GPU_OUT

for g in $(seq 0 $((NUM_GPUS-1))); do
    GPU_PIDS[$g]=""
    GPU_DESC[$g]=""
    GPU_OUT[$g]=""
done

total_done=0
total_failed=0
job_idx=0

gpu_memory_free_mb() {
    # Returns free memory in MiB for a given GPU index
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits         --id="$1" 2>/dev/null | tr -d ' '
}

wait_for_free_gpu() {
    while true; do
        for gpu in $(seq 0 $((NUM_GPUS-1))); do
            pid="${GPU_PIDS[$gpu]}"
            if [[ -z "$pid" ]]; then
                # GPU slot is unregistered — but check actual GPU memory
                # to guard against orphaned processes from crashed jobs
                free_mb=$(gpu_memory_free_mb "$gpu")
                if [[ -n "$free_mb" ]] && [[ "$free_mb" -gt 20000 ]]; then
                    # >20 GiB free — safe to launch
                    echo "$gpu"; return
                fi
                # GPU slot is free in our tracking but memory is still occupied
                # (previous job left orphaned CUDA context). Wait and retry.
                continue
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null || true
                local out_key="${GPU_OUT[$gpu]:-}"
                if [[ -n "$out_key" ]] && compgen -G "${out_key}/**/summary.csv" > /dev/null 2>&1; then
                    echo "[done ] GPU=${gpu} ${GPU_DESC[$gpu]}" >&2
                else
                    echo "[FAIL ] GPU=${gpu} ${GPU_DESC[$gpu]} (no summary.csv)" >&2
                    total_failed=$((total_failed+1))
                fi
                total_done=$((total_done+1))
                echo "[queue] ${total_done}/${TOTAL} done | ${total_failed} failed" >&2
                GPU_PIDS[$gpu]=""
                GPU_OUT[$gpu]=""
                # Wait for GPU memory to actually free before declaring available
                echo "[wait ] GPU=${gpu} waiting for memory to release..." >&2
                for _ in $(seq 1 10); do
                    sleep 3
                    free_mb=$(gpu_memory_free_mb "$gpu")
                    if [[ -n "$free_mb" ]] && [[ "$free_mb" -gt 20000 ]]; then
                        break
                    fi
                done
                echo "$gpu"; return
            fi
        done
        sleep 3
    done
}

launch_job() {
    local gpu=$1
    local jj=$2

    # Parse all fields — pipe JSON to the pre-written parser file.
    # This avoids the heredoc-steals-stdin bug where
    # eval "$(echo ... | python - << 'PY' ... PY)" drops the piped input.
    eval "$(echo "${jj}" | python "${JOB_PARSER}")"

    [[ -f "${jsonl}" ]] || { echo "[WARN ] input not found: ${jsonl}" >&2; return; }

    local gpu_mem_util="0.92"
    [[ "${method}" == "draft_sd" ]] && gpu_mem_util="0.80"

    local desc="${exp}/${workload}/${method}/k${k}/T${temp}/ctx${ctx}/t${turns}"
    GPU_DESC[$gpu]="${desc}"
    GPU_OUT[$gpu]="${outdir}/${exp}/${workload}/${method}"
    local log="${LOG_DIR}/${exp}_${workload}_${method}_k${k}_T${temp}_ctx${ctx}_t${turns}.log"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${BENCHMARK}" \
        --experiment            "${exp}" \
        --workload              "${workload}" \
        --run_name              "${workload}" \
        --method                "${method}" \
        --k                     "${k}" \
        --temperature           "${temp}" \
        --ngram_lookup_min      "${nlmin}" \
        --ngram_lookup_max      "${nlmax}" \
        --draft_model           "${draft}" \
        --eagle3_model          "${eagle}" \
        --input_jsonl           "${jsonl}" \
        --max_tokens            "${maxtok}" \
        --max_prompt_tokens     "${ctx}" \
        --num_turns             "${turns}" \
        --limit                 "${limit}" \
        --out_dir               "${outdir}" \
        --model                 "${model}" \
        --seed                  42 \
        --tensor_parallel_size  1 \
        --gpu_memory_utilization "${gpu_mem_util}" \
        > "${log}" 2>&1 &

    GPU_PIDS[$gpu]=$!
}

echo "[study] Starting queue..."
while IFS= read -r job_json; do
    gpu=$(wait_for_free_gpu)
    job_idx=$((job_idx+1))
    desc=$(echo "$job_json" | python -c "
import sys,json
j=json.loads(sys.stdin.read())
dm=(' draft='+j['draft_model'].split('/')[-1]) if j['draft_model'] else ''
print(f\"{j['experiment']:32s} {j['workload']:28s} {j['method']:10s} k={j['k']} T={j['temperature']} ctx={j['max_prompt_tokens']} t={j['num_turns']}{dm}\")
")
    echo "[launch] ${job_idx}/${TOTAL} GPU=${gpu} ${desc}"
    launch_job "$gpu" "$job_json"
done < "${JOBS_FILE}"

# ── Drain ──────────────────────────────────────────────────────────────────
echo "[study] Draining final jobs..."
for gpu in $(seq 0 $((NUM_GPUS-1))); do
    pid="${GPU_PIDS[$gpu]}"
    if [[ -n "$pid" ]]; then
        wait "$pid" 2>/dev/null || true
        local_out="${GPU_OUT[$gpu]:-}"
        if [[ -n "$local_out" ]] && compgen -G "${local_out}/**/summary.csv" > /dev/null 2>&1; then
            echo "[done ] GPU=${gpu} ${GPU_DESC[$gpu]}"
        else
            echo "[FAIL ] GPU=${gpu} ${GPU_DESC[$gpu]} (no summary.csv)"
            total_failed=$((total_failed+1))
        fi
        total_done=$((total_done+1))
    fi
done

[[ "${OWNS_JOBS_FILE}" -eq 1 ]] && rm -f "${JOBS_FILE}"
rm -f "${JOB_PARSER}" 2>/dev/null || true

echo ""
echo "=================================================="
echo "[study] COMPLETE"
echo "  Total generated : ${ALL_JOBS}"
echo "  Ran this session: ${TOTAL}"
echo "  Done            : ${total_done}"
echo "  Failed          : ${total_failed}"
echo "  Config          : ${EXP_CONFIG}"
echo "  Logs            : ${LOG_DIR}"
echo "=================================================="
[[ "${total_failed}" -eq 0 ]] || exit 1