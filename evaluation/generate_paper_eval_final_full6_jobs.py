#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path("/fsx/jmanvi/Internship_project/ASD")

# Repo folder is paper_eval_fin in GitHub.
SOURCE_CANDIDATES = [
    ROOT / "slurm_logs/paper_eval_fin",
    ROOT / "paper_eval_fin",
]

EVAL_ROOT = ROOT / "results/apexp_block/config_large_500/evaluation/paper_eval_final_full6"
SLURM_ROOT = ROOT / "slurm_logs/paper_eval_final_full6"

TEST_FILES = {
    "code_gen": ROOT / "data/benchmarks/By_split_phase_1/code_gen_test.jsonl",
    "conversational_generation": ROOT / "data/benchmarks/By_split_phase_1/conversational_generation_test.jsonl",
    "long_chain_reasoning": ROOT / "data/benchmarks/By_split_phase_1/long_chain_reasoning_test.jsonl",
    "long_context_completion": ROOT / "data/benchmarks/By_split_phase_1/long_context_completion_test.jsonl",
    "long_horizon_swe": ROOT / "data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl",
    "mathematical_reasoning": ROOT / "data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl",
}

MAX_TOKENS = {
    "code_gen": 512,
    "conversational_generation": 512,
    "long_chain_reasoning": 1024,
    "long_context_completion": 2048,
    "long_horizon_swe": 1024,
    "mathematical_reasoning": 512,
}

BASELINE_METHODS = ["ngram_sd", "draft_sd", "eagle3"]
BASELINE_KS = [1, 2, 4, 8, 16]


def find_source_root() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists() and (p / "apex").exists() and (p / "baselines").exists() and (p / "ar").exists():
            return p
    raise SystemExit(
        "Could not find paper_eval_final or paper_eval_fin with apex/baselines/ar folders."
    )


def patch_sbatch_header(txt: str, job_kind: str, stem: str, gres: str, cpus: int, mem: str) -> str:
    lines = txt.splitlines()
    out = []
    seen_partition = False
    seen_gres = False
    seen_cpus = False
    seen_mem = False
    seen_output = False
    seen_error = False

    for line in lines:
        if line.startswith("#SBATCH --partition") or line.startswith("#SBATCH -p"):
            out.append("#SBATCH --partition=dev")
            seen_partition = True
            continue
        if line.startswith("#SBATCH --time") or line.startswith("#SBATCH -t"):
            continue
        if line.startswith("#SBATCH --exclusive"):
            continue
        if line.startswith("#SBATCH --gres"):
            out.append(f"#SBATCH --gres={gres}")
            seen_gres = True
            continue
        if line.startswith("#SBATCH --gpus-per-node"):
            continue
        if line.startswith("#SBATCH --cpus-per-task"):
            out.append(f"#SBATCH --cpus-per-task={cpus}")
            seen_cpus = True
            continue
        if line.startswith("#SBATCH --mem"):
            out.append(f"#SBATCH --mem={mem}")
            seen_mem = True
            continue
        if line.startswith("#SBATCH --output"):
            out.append(f"#SBATCH --output={SLURM_ROOT / job_kind / (stem + '_%j.out')}")
            seen_output = True
            continue
        if line.startswith("#SBATCH --error"):
            out.append(f"#SBATCH --error={SLURM_ROOT / job_kind / (stem + '_%j.err')}")
            seen_error = True
            continue
        out.append(line)

    txt = "\n".join(out) + "\n"

    insert = []
    if not seen_partition:
        insert.append("#SBATCH --partition=dev")
    if not seen_gres:
        insert.append(f"#SBATCH --gres={gres}")
    if not seen_cpus:
        insert.append(f"#SBATCH --cpus-per-task={cpus}")
    if not seen_mem:
        insert.append(f"#SBATCH --mem={mem}")
    if not seen_output:
        insert.append(f"#SBATCH --output={SLURM_ROOT / job_kind / (stem + '_%j.out')}")
    if not seen_error:
        insert.append(f"#SBATCH --error={SLURM_ROOT / job_kind / (stem + '_%j.err')}")

    if insert:
        txt = txt.replace("#!/bin/bash\n", "#!/bin/bash\n" + "\n".join(insert) + "\n", 1)

    return txt


def replace_apex_input_block(txt: str) -> str:
    files = "\n".join([f"    {p} \\" for p in TEST_FILES.values()])
    new_block = "--input-jsonl \\\n" + files.rstrip(" \\") + " \\\n  --gpu-memory-utilization 0.60"

    # Replace from --input-jsonl through --gpu-memory-utilization value.
    txt = re.sub(
        r"--input-jsonl\s+\\\n(?:\s+.*?\.jsonl\s+\\\n)+\s+--gpu-memory-utilization\s+[0-9.]+",
        new_block,
        txt,
        flags=re.DOTALL,
    )

    return txt


def patch_apex_sbatch(src: Path, dst: Path):
    stem = src.stem
    txt = src.read_text()

    txt = patch_sbatch_header(
        txt,
        job_kind="apex",
        stem=stem,
        gres="gpu:6",
        cpus=48,
        mem="420G",
    )

    # Fresh output root.
    txt = re.sub(
        r'export OUT_DIR="[^"]+"',
        f'export OUT_DIR="{EVAL_ROOT / "apex_hot_runs" / stem}"',
        txt,
    )

    # Replace old cap50 input files with the new six full test files.
    txt = replace_apex_input_block(txt)

    # Make sure no old hardware or split conversational cap50 remains.
    txt = txt.replace("paper_cap50_profile_kgrid", "paper_eval_final_full6")
    txt = txt.replace("paper_cap50", "paper_eval_final_full6")

    # Use dev-friendly memory util.
    txt = re.sub(r"--gpu-memory-utilization\s+[0-9.]+", "--gpu-memory-utilization 0.60", txt)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(txt)
    dst.chmod(0o755)


def write_baseline_sbatch(method: str, k: int, dst: Path):
    stem = f"base_{method}_k{k}"
    out_root = EVAL_ROOT / "fixed_runs"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={stem}",
        "#SBATCH --partition=dev",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=180G",
        f"#SBATCH --output={SLURM_ROOT / 'baselines' / (stem + '_%j.out')}",
        f"#SBATCH --error={SLURM_ROOT / 'baselines' / (stem + '_%j.err')}",
        "",
        "set -euo pipefail",
        "",
        "cd /fsx/jmanvi/Internship_project/ASD",
        "source /fsx/jmanvi/Internship_project/.venv/bin/activate",
        "",
        'export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"',
        "export PYTHONUNBUFFERED=1",
        "export TOKENIZERS_PARALLELISM=false",
        "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "",
        "export CUDA_HOME=/usr/local/cuda-12.9",
        "export CUDA_PATH=/usr/local/cuda-12.9",
        "export PATH=/usr/local/cuda-12.9/bin:$PATH",
        'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}',
        "export NVCC=/usr/local/cuda-12.9/bin/nvcc",
        "",
        f'echo "[BASELINE FINAL6] method={method} k={k}"',
        'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"',
        "nvidia-smi || true",
        "",
    ]

    for workload, input_path in TEST_FILES.items():
        lines += [
            f'echo "[BASELINE FINAL6] method={method} k={k} workload={workload}"',
            "",
            "python3 src/layer0_data_collection/run_benchmark.py \\",
            f'  --workload "{workload}" \\',
            f'  --run_name "{workload}" \\',
            '  --experiment "paper_eval_final_full6_fixed" \\',
            f'  --input_jsonl "{input_path}" \\',
            "  --limit 0 \\",
            f'  --out_dir "{out_root}" \\',
            '  --model "Qwen/Qwen3-8B" \\',
            '  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \\',
            '  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \\',
            f'  --method "{method}" \\',
            f"  --k {k} \\",
            f"  --max_tokens {MAX_TOKENS[workload]} \\",
            "  --temperature 0.0 \\",
            "  --seed 42 \\",
            "  --enforce_eager \\",
            "  --gpu_memory_utilization 0.60",
            "",
        ]

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")
    dst.chmod(0o755)


def write_ar_sbatch(workload: str, dst: Path):
    stem = f"ar_{workload}"
    out_dir = EVAL_ROOT / "fixed_runs" / workload / "ar" / "ar_temp0.0"

    txt = f"""#!/bin/bash
#SBATCH --job-name={stem}
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output={SLURM_ROOT / 'ar' / (stem + '_%j.out')}
#SBATCH --error={SLURM_ROOT / 'ar' / (stem + '_%j.err')}

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_HOME=/usr/local/cuda-12.9
export CUDA_PATH=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${{LD_LIBRARY_PATH:-}}
export NVCC=/usr/local/cuda-12.9/bin/nvcc

mkdir -p "{out_dir}"

echo "[AR FINAL6]"
echo "workload={workload}"
echo "input_jsonl={TEST_FILES[workload]}"
echo "out_dir={out_dir}"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-UNSET}}"
nvidia-smi || true

python3 src/layer0_data_collection/run_benchmark.py \\
  --workload "{workload}" \\
  --input_jsonl "{TEST_FILES[workload]}" \\
  --out_dir "{out_dir}" \\
  --experiment "paper_eval_final_full6_ar" \\
  --run_name "{workload}_full" \\
  --method ar \\
  --k 1 \\
  --limit 0 \\
  --temperature 0.0 \\
  --max_tokens {MAX_TOKENS[workload]} \\
  --seed 42 \\
  --enforce_eager \\
  --gpu_memory_utilization 0.80 \\
  --model "Qwen/Qwen3-8B"

echo "[done] {workload}"
"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(txt)
    dst.chmod(0o755)


def write_throttled_submit(job_paths: list[Path]):
    job_list = SLURM_ROOT / "submit_all_jobs.txt"
    job_list.parent.mkdir(parents=True, exist_ok=True)
    job_list.write_text("\n".join(str(p) for p in job_paths) + "\n")

    submitter = SLURM_ROOT / "submit_throttled.sh"
    submitter.write_text("""#!/bin/bash
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
""")
    submitter.chmod(0o755)

    return job_list, submitter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_root = find_source_root()

    for workload, p in TEST_FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing test file for {workload}: {p}")

    if SLURM_ROOT.exists():
        if not args.force:
            raise SystemExit(f"{SLURM_ROOT} already exists. Re-run with --force to overwrite generated slurms.")
        shutil.rmtree(SLURM_ROOT)

    SLURM_ROOT.mkdir(parents=True, exist_ok=True)
    (SLURM_ROOT / "apex").mkdir(parents=True, exist_ok=True)
    (SLURM_ROOT / "baselines").mkdir(parents=True, exist_ok=True)
    (SLURM_ROOT / "ar").mkdir(parents=True, exist_ok=True)

    job_paths = []

    # APEX: patch every actual APEX sbatch in the repo folder.
    apex_srcs = sorted(
        p for p in (src_root / "apex").glob("*.sbatch")
        if p.is_file()
    )

    for src in apex_srcs:
        dst = SLURM_ROOT / "apex" / src.name
        patch_apex_sbatch(src, dst)
        job_paths.append(dst)

    # Fixed speculative baselines.
    for method in BASELINE_METHODS:
        for k in BASELINE_KS:
            dst = SLURM_ROOT / "baselines" / f"base_{method}_k{k}.sbatch"
            write_baseline_sbatch(method, k, dst)
            job_paths.append(dst)

    # AR baselines: one job per workload.
    for workload in TEST_FILES:
        dst = SLURM_ROOT / "ar" / f"ar_{workload}.sbatch"
        write_ar_sbatch(workload, dst)
        job_paths.append(dst)

    job_list, submitter = write_throttled_submit(job_paths)

    print("SOURCE_ROOT:", src_root)
    print("EVAL_ROOT:", EVAL_ROOT)
    print("SLURM_ROOT:", SLURM_ROOT)
    print("APEX_JOBS:", len(apex_srcs))
    print("BASELINE_JOBS:", len(BASELINE_METHODS) * len(BASELINE_KS))
    print("AR_JOBS:", len(TEST_FILES))
    print("TOTAL_JOBS:", len(job_paths))
    print("JOB_LIST:", job_list)
    print("SUBMITTER:", submitter)
    print()
    print("Submit with:")
    print(f"  MAX_ACTIVE=10 bash {submitter}")


if __name__ == "__main__":
    main()
