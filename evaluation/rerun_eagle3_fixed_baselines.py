#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


WORKLOADS = [
    {
        "workload": "code_gen",
        "input_jsonl": "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_test.jsonl",
        "max_tokens": 512,
    },
    {
        "workload": "conversational_generation_sft",
        "input_jsonl": "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl",
        "max_tokens": 512,
    },
    {
        "workload": "conversational_generation_gen",
        "input_jsonl": "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl",
        "max_tokens": 512,
    },
    {
        "workload": "long_chain_reasoning",
        "input_jsonl": "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning_synthetic.jsonl",
        "max_tokens": 1024,
    },
    {
        "workload": "long_context_completion",
        "input_jsonl": "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion_synthetic.jsonl",
        "max_tokens": 2048,
    },
    {
        "workload": "long_horizon_swe",
        "input_jsonl": "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl",
        "max_tokens": 1024,
    },
    {
        "workload": "mathematical_reasoning",
        "input_jsonl": "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl",
        "max_tokens": 512,
    },
]

KS = [1, 2, 4, 8, 16]

DEFAULT_ROOT = Path("/fsx/jmanvi/Internship_project/ASD")
FIXED_RUNS_ROOT = DEFAULT_ROOT / "results/apexp_block/config_large_500/evaluation/final_evaluations/fixed_runs"
TMP_ROOT = DEFAULT_ROOT / "results/apexp_block/config_large_500/evaluation/final_evaluations/eagle3_rerun_tmp"
LOG_ROOT = DEFAULT_ROOT / "slurm_logs/rerun_eagle3_fixed"


def exact_target_dir(workload: str, k: int) -> Path:
    return (
        FIXED_RUNS_ROOT
        / workload
        / "eagle3"
        / f"k{k}_temp0.0_eagle_Qwen3_8B_speculator.eagle3"
    )


def make_manifest(path: Path):
    rows = []
    for w in WORKLOADS:
        for k in KS:
            target = exact_target_dir(w["workload"], k)
            tmp = TMP_ROOT / w["workload"] / "eagle3" / f"k{k}"
            rows.append({
                "workload": w["workload"],
                "input_jsonl": w["input_jsonl"],
                "k": k,
                "max_tokens": w["max_tokens"],
                "target_dir": str(target),
                "tmp_dir": str(tmp),
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    return rows


def read_manifest(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_leaf_output_dir(tmp_dir: Path) -> Path:
    summaries = sorted(tmp_dir.rglob("summary.csv"))
    if not summaries:
        raise RuntimeError(f"No summary.csv found under tmp_dir={tmp_dir}")

    # Prefer the deepest summary.csv, because run_benchmark may create nested dirs.
    summaries = sorted(summaries, key=lambda p: len(p.parts), reverse=True)
    return summaries[0].parent


def copy_leaf_to_target(leaf: Path, target: Path, metadata: dict):
    if target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)

    for item in leaf.iterdir():
        dst = target / item.name
        if item.is_file():
            shutil.copy2(item, dst)
        elif item.is_dir():
            shutil.copytree(item, dst)

    (target / "_eagle3_rerun_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def run_one(row: dict):
    root = DEFAULT_ROOT
    tmp_dir = Path(row["tmp_dir"])
    target_dir = Path(row["target_dir"])

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{root}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"

    # Use cached models if the node has cache. Remove these two exports if a model needs downloading.
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")

    cmd = [
        "python3",
        "src/layer0_data_collection/run_benchmark.py",
        "--workload", row["workload"],
        "--run_name", f"{row['workload']}_eagle3_k{row['k']}_rerun",
        "--experiment", "baseline_matrix_all_methods_all_k",
        "--input_jsonl", row["input_jsonl"],
        "--out_dir", str(tmp_dir),
        "--method", "eagle3",
        "--k", str(row["k"]),
        "--model", "Qwen/Qwen3-8B",
        "--eagle3_model", "RedHatAI/Qwen3-8B-speculator.eagle3",
        "--temperature", "0.0",
        "--top_p", "1.0",
        "--max_tokens", str(row["max_tokens"]),
        "--limit", "0",
        "--gpu_memory_utilization", os.environ.get("APEXP_RERUN_GPU_MEM", "0.60"),
        "--enforce_eager",
        "--seed", "42",
    ]

    log_file = tmp_dir / "rerun_command.log"
    start = time.time()

    print("============================================================", flush=True)
    print("[EAGLE3 RERUN]", json.dumps(row, indent=2), flush=True)
    print("[CMD]", " ".join(cmd), flush=True)
    print("============================================================", flush=True)

    with log_file.open("w") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )

    end = time.time()

    if proc.returncode != 0:
        raise RuntimeError(
            f"run_benchmark failed for workload={row['workload']} k={row['k']} "
            f"returncode={proc.returncode}. See {log_file}"
        )

    leaf = find_leaf_output_dir(tmp_dir)

    metadata = {
        **row,
        "event": "eagle3_fixed_baseline_rerun",
        "time_start": start,
        "time_end": end,
        "elapsed_s": end - start,
        "cmd": cmd,
        "tmp_dir": str(tmp_dir),
        "leaf_output_dir": str(leaf),
        "target_dir": str(target_dir),
        "note": "Target directory was overwritten by this rerun.",
    }

    copy_leaf_to_target(leaf, target_dir, metadata)

    print("[DONE]", row["workload"], "k", row["k"], flush=True)
    print("[LEAF]", leaf, flush=True)
    print("[TARGET]", target_dir, flush=True)


def write_sbatch(manifest: Path, sbatch_path: Path, n_jobs: int, max_parallel: int):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    text = f"""#!/bin/bash
#SBATCH --job-name=eagle3_fixed_rerun
#SBATCH --partition=ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-{n_jobs - 1}%{max_parallel}
#SBATCH --output={LOG_ROOT}/eagle3_fixed_rerun_%A_%a.out
#SBATCH --error={LOG_ROOT}/eagle3_fixed_rerun_%A_%a.err

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

# Use cached models. Comment these out if the model needs downloading.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-UNSET}}"
nvidia-smi || true

python3 evaluation/rerun_eagle3_fixed_baselines.py \\
  --mode run-one \\
  --manifest "{manifest}" \\
  --index "${{SLURM_ARRAY_TASK_ID}}"
"""
    sbatch_path.parent.mkdir(parents=True, exist_ok=True)
    sbatch_path.write_text(text)
    sbatch_path.chmod(0o755)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prepare", "run-one"], required=True)
    ap.add_argument(
        "--manifest",
        default=str(DEFAULT_ROOT / "results/apexp_block/config_large_500/evaluation/final_evaluations/eagle3_rerun_manifest.jsonl"),
    )
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--max-parallel", type=int, default=4)
    args = ap.parse_args()

    manifest = Path(args.manifest)

    if args.mode == "prepare":
        rows = make_manifest(manifest)
        sbatch_path = LOG_ROOT / "run_eagle3_fixed_rerun_array.sbatch"
        write_sbatch(manifest, sbatch_path, n_jobs=len(rows), max_parallel=args.max_parallel)

        print("WROTE manifest:", manifest)
        print("WROTE sbatch:", sbatch_path)
        print("N jobs:", len(rows))
        print("Submit:")
        print(f"  sbatch {sbatch_path}")
        return

    rows = read_manifest(manifest)
    if args.index < 0 or args.index >= len(rows):
        raise SystemExit(f"Bad index={args.index}; manifest has {len(rows)} rows")

    run_one(rows[args.index])


if __name__ == "__main__":
    main()
