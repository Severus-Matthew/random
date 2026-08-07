#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.apex_eval_utils import ensure_dir, slugify


DEFAULT_SLOW_GPUS = "ngram_sd:0,eagle3:1,draft_sd:2"
DEFAULT_FAST_GPUS = "ngram_sd:3,eagle3:4,draft_sd:5"


def parse_candidate_sets(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"candidate set must be name=csv, got {item}")
        name, csv = item.split("=", 1)
        vals = [str(int(x.strip())) for x in csv.split(",") if x.strip()]
        if not vals:
            raise ValueError(f"empty candidate set: {item}")
        out.append((slugify(name), ",".join(vals)))
    return out


def temp_label(t: float) -> str:
    return f"temp{str(t).replace('.', 'p').replace('-', 'm')}"


def model_label(path: str) -> str:
    p = Path(path)
    # Include parent for v4/v4b disambiguation.
    return slugify(f"{p.parent.name}_{p.name}")


def sbatch_text(
    *,
    job_name: str,
    out_dir: Path,
    slow_router_dir: str,
    model_dir: str,
    workload_files: list[str],
    temperature: float,
    candidate_csv: str,
    min_k: int,
    max_k: int,
    max_prompts_per_file: int,
    slow_gpus: str,
    fast_gpus: str,
    gpu_memory_utilization: float,
    time_limit: str,
    cpus: int,
    mem: str,
    partition: str,
    gpus: int,
    target_model: str,
    draft_model: str,
    eagle3_model: str,
    device: str,
    every_n: int,
    switch_margin: float,
):
    input_lines = " \\\n    ".join(workload_files)
    cap_arg = f"--max-prompts-per-file {int(max_prompts_per_file)}" if int(max_prompts_per_file) > 0 else "--max-prompts-per-file 0"
    return f'''#!/bin/bash
#SBATCH --job-name={job_name[:48]}
#SBATCH --output=slurm_logs/evaluation_jobs/{job_name}_%j.out
#SBATCH --error=slurm_logs/evaluation_jobs/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export APEXP_ONLINE_FAST=1
export APEXP_LEARNED_FAST_MODEL_DIR="{model_dir}"
export APEXP_LEARNED_FAST_DEVICE={device}
export APEXP_LEARNED_FAST_EVERY_N={every_n}
export APEXP_LEARNED_FAST_SWITCH_MARGIN={switch_margin}
export APEXP_LEARNED_FAST_TRACE_SCORES=0

export APEXP_ONLINE_FAST_TRACE=1
export APEXP_FAST_CANDIDATE_KS="{candidate_csv}"
export APEXP_MIN_K={min_k}
export APEXP_MAX_K={max_k}
export APEXP_FAST_WINDOW=2
export APEXP_FAST_MIN_OBS=2

export OUT_DIR="{out_dir}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 src/apexp/week2/realtime_hot_e2e.py \
  --slow-router-dir "{slow_router_dir}" \
  --out-dir "$OUT_DIR" \
  {cap_arg} \
  --fast-candidate-ks "{candidate_csv}" \
  --fast-window 2 \
  --fast-min-obs 2 \
  --slow-gpus "{slow_gpus}" \
  --fast-gpus "{fast_gpus}" \
  --target-model "{target_model}" \
  --draft-model "{draft_model}" \
  --eagle3-model "{eagle3_model}" \
  --input-jsonl \
    {input_lines} \
  --gpu-memory-utilization {gpu_memory_utilization} \
  2>&1 | tee "$OUT_DIR/run.log"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Where to write sbatch files")
    ap.add_argument("--result-root", required=True, help="Root for evaluation run outputs")
    ap.add_argument("--slow-router-dir", required=True)
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--workload-files", nargs="+", required=True)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.1, 0.5])
    ap.add_argument("--candidate-sets", nargs="+", default=["k8_16=8,16", "k4_8_16=4,8,16", "k1_16=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"])
    ap.add_argument("--max-prompts-per-file", type=int, default=20)
    ap.add_argument("--slow-gpus", default=DEFAULT_SLOW_GPUS)
    ap.add_argument("--fast-gpus", default=DEFAULT_FAST_GPUS)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--cpus", type=int, default=48)
    ap.add_argument("--mem", default="420G")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--gpus", type=int, default=6)
    ap.add_argument("--target-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eagle3-model", default="RedHatAI/Qwen3-8B-speculator.eagle3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--switch-margin", type=float, default=0.03)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    ensure_dir("slurm_logs/evaluation_jobs")
    result_root = Path(args.result_root)
    candidate_sets = parse_candidate_sets(args.candidate_sets)

    written = []
    for model_dir in args.model_dirs:
        if not Path(model_dir).exists():
            print(f"[warn] model dir does not exist yet, still generating job: {model_dir}")
        mlabel = model_label(model_dir)
        for t in args.temps:
            tlab = temp_label(t)
            for cname, ccsv in candidate_sets:
                vals = [int(x) for x in ccsv.split(",")]
                min_k, max_k = min(vals), max(vals)
                run_name = f"{mlabel}__{tlab}__{cname}"
                job_name = f"apex_eval_{run_name}"[:60]
                run_out = result_root / run_name
                text = sbatch_text(
                    job_name=job_name,
                    out_dir=run_out,
                    slow_router_dir=args.slow_router_dir,
                    model_dir=model_dir,
                    workload_files=args.workload_files,
                    temperature=t,
                    candidate_csv=ccsv,
                    min_k=min_k,
                    max_k=max_k,
                    max_prompts_per_file=args.max_prompts_per_file,
                    slow_gpus=args.slow_gpus,
                    fast_gpus=args.fast_gpus,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    time_limit=args.time,
                    cpus=args.cpus,
                    mem=args.mem,
                    partition=args.partition,
                    gpus=args.gpus,
                    target_model=args.target_model,
                    draft_model=args.draft_model,
                    eagle3_model=args.eagle3_model,
                    device=args.device,
                    every_n=args.every_n,
                    switch_margin=args.switch_margin,
                )
                path = out_dir / f"{run_name}.sbatch"
                path.write_text(text)
                written.append(path)

    submit_all = out_dir / "submit_all.sh"
    submit_all.write_text("#!/bin/bash\nset -euo pipefail\n" + "\n".join(f"sbatch {p}" for p in written) + "\n")
    print(f"WROTE {len(written)} sbatch files to {out_dir}")
    print(f"Submit all: bash {submit_all}")


if __name__ == "__main__":
    main()
