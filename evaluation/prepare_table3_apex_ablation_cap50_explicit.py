#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_DIR = Path("/fsx/jmanvi/Internship_project/ASD")

WORKLOADS = [
    {
        "workload": "code_gen",
        "input_jsonl": str(ROOT_DIR / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/code_gen_cap50.jsonl"),
        "max_tokens": 512,
    },
    {
        "workload": "conversational_generation_sft",
        "input_jsonl": str(ROOT_DIR / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_sft_cap50.jsonl"),
        "max_tokens": 512,
    },
    {
        "workload": "conversational_generation_gen",
        "input_jsonl": str(ROOT_DIR / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_gen_cap50.jsonl"),
        "max_tokens": 512,
    },
    {
        "workload": "long_chain_reasoning",
        "input_jsonl": str(ROOT_DIR / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_chain_reasoning_cap50.jsonl"),
        "max_tokens": 1024,
    },
    {
        "workload": "long_context_completion",
        "input_jsonl": str(ROOT_DIR / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_context_completion_cap50.jsonl"),
        "max_tokens": 2048,
    },
    {
        "workload": "long_horizon_swe",
        "input_jsonl": str(ROOT_DIR / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_horizon_swe_cap50.jsonl"),
        "max_tokens": 1024,
    },
    {
        "workload": "mathematical_reasoning",
        "input_jsonl": str(ROOT_DIR / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl"),
        "max_tokens": 512,
    },
]

PARTITION = "ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40"

BASE_MODEL_ROOT = Path("results/apexp_block/config_large_500/week2/survival_utility_controller_v4")
NO_SURV_ROOT = Path("results/apexp_block/config_large_500/week2/ablation_no_survival_loss_v4")

SLOW_ROUTER_DIR = "results/apexp_block/config_large_500/week2/request_prompt_router_d128"

ANCHORS = {
    "APEX_B": {
        "paper_name": "APEX-B",
        "base_model_dir": str(BASE_MODEL_ROOT / "balanced"),
        "no_survival_model_dir": str(NO_SURV_ROOT / "APEX_B_no_survival_loss"),
    },
    "APEX_S": {
        "paper_name": "APEX-S",
        "base_model_dir": str(BASE_MODEL_ROOT / "speed_first"),
        "no_survival_model_dir": str(NO_SURV_ROOT / "APEX_S_no_survival_loss"),
    },
    "APEX_E": {
        "paper_name": "APEX-E",
        "base_model_dir": str(BASE_MODEL_ROOT / "efficient"),
        "no_survival_model_dir": str(NO_SURV_ROOT / "APEX_E_no_survival_loss"),
    },
}

QUALITY_BALANCED_ENV = {
    "APEXP_LEARNED_FAST_ASYNC": "1",
    "APEXP_LEARNED_FAST_EVERY_N": "4",
    "APEXP_LEARNED_FAST_FIRST_WINDOW": "3",
    "APEXP_FAST_WINDOW": "3",
    "APEXP_FAST_MIN_OBS": "3",

    "APEXP_ONLINE_SCORE_UTILITY_WEIGHT": "0.7",
    "APEXP_ONLINE_SCORE_ACCEPT_WEIGHT": "0.22",
    "APEXP_ONLINE_SCORE_ACCEPT_RATE_WEIGHT": "0.75",
    "APEXP_ONLINE_SCORE_TPS_WEIGHT": "0.25",
    "APEXP_ONLINE_SCORE_WASTE_WEIGHT": "0.80",
    "APEXP_ONLINE_SCORE_BLOCK_PENALTY_WEIGHT": "0.12",
    "APEXP_ONLINE_MIN_EXPECTED_ACCEPTED": "0.80",
    "APEXP_ONLINE_MIN_ACCEPT_RATE": "0.12",
    "APEXP_ONLINE_BAD_ACTION_PENALTY": "4.0",

    "APEXP_ABLATE_ENTROPY": "0",
    "APEXP_ABLATE_REPETITION": "0",
    "APEXP_ABLATE_VERIFIER_HISTORY": "0",
}

ABLATIONS = [
    {
        "name": "base",
        "paper_variant": "Full APEX",
        "type": "reference",
        "model": "base",
        "env": {},
    },
    {
        "name": "no_survival_loss",
        "paper_variant": "w/o survival loss",
        "type": "training_loss",
        "model": "no_survival",
        "env": {},
    },
    {
        "name": "no_cost_head",
        "paper_variant": "w/o cost head",
        "type": "runtime_score",
        "model": "base",
        "env": {
            "APEXP_ONLINE_SCORE_TPS_WEIGHT": "0.0",
        },
    },
    {
        "name": "no_entropy_features",
        "paper_variant": "w/o entropy features",
        "type": "feature_mask",
        "model": "base",
        "env": {
            "APEXP_ABLATE_ENTROPY": "1",
        },
    },
    {
        "name": "no_repetition_features",
        "paper_variant": "w/o repetition features",
        "type": "feature_mask",
        "model": "base",
        "env": {
            "APEXP_ABLATE_REPETITION": "1",
        },
    },
    {
        "name": "no_verifier_history_features",
        "paper_variant": "w/o verifier-history features",
        "type": "feature_mask",
        "model": "base",
        "env": {
            "APEXP_ABLATE_VERIFIER_HISTORY": "1",
        },
    },
]


def sbatch_text(
    *,
    run_name: str,
    run_dir: Path,
    model_dir: str,
    anchor: str,
    paper_anchor: str,
    ablation: dict,
    log_dir: Path,
):
    input_lines = " \\\n    ".join([w["input_jsonl"] for w in WORKLOADS])

    env = dict(QUALITY_BALANCED_ENV)
    env.update(ablation["env"])

    env_lines = "\n".join([f'export {k}="{v}"' for k, v in sorted(env.items())])

    return f'''#!/bin/bash
#SBATCH --job-name=t3_{anchor.lower()}_{ablation["name"][:18]}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=700G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output={log_dir / (run_name + ".out")}
#SBATCH --error={log_dir / (run_name + ".err")}

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
source /fsx/jmanvi/Internship_project/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export OUT_DIR="{run_dir}"
export APEXP_ONLINE_FAST=1
export APEXP_LEARNED_FAST_MODEL_DIR="{model_dir}"
export APEXP_LEARNED_FAST_DEVICE="cuda"
export APEXP_LEARNED_FAST_TRACE_SCORES=0

export APEXP_TABLE3_ANCHOR="{anchor}"
export APEXP_TABLE3_PAPER_ANCHOR="{paper_anchor}"
export APEXP_TABLE3_ABLATION="{ablation["name"]}"
export APEXP_TABLE3_VARIANT="{ablation["paper_variant"]}"

export APEXP_FAST_CANDIDATE_KS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
export APEXP_MIN_K=1
export APEXP_MAX_K=16

{env_lines}

echo "============================================================"
echo "TABLE 3 APEX ABLATION"
echo "RUN_NAME={run_name}"
echo "OUT_DIR=$OUT_DIR"
echo "ANCHOR=$APEXP_TABLE3_ANCHOR"
echo "VARIANT=$APEXP_TABLE3_VARIANT"
echo "MODEL_DIR=$APEXP_LEARNED_FAST_MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-UNSET}}"
echo "============================================================"
env | grep -E 'APEXP_(TABLE3|ONLINE_SCORE|ONLINE_MIN|LEARNED_FAST|FAST_|ABLATE)' | sort
nvidia-smi || true

if [ -e "$OUT_DIR" ]; then
  echo "[ERROR] OUT_DIR already exists. Refusing to overwrite for reproducibility:"
  echo "$OUT_DIR"
  echo "Move/delete it manually only if you intentionally want to rerun this exact config."
  exit 44
fi

mkdir -p "$OUT_DIR"

python3 src/apexp/week2/realtime_hot_e2e.py \\
  --slow-router-dir "{SLOW_ROUTER_DIR}" \\
  --out-dir "$OUT_DIR" \\
  --max-prompts-per-file 0 \\
  --fast-candidate-ks "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" \\
  --fast-min-k 1 \\
  --fast-window 3 \\
  --fast-min-obs 3 \\
  --slow-gpus "ngram_sd:0,eagle3:1,draft_sd:2" \\
  --fast-gpus "ngram_sd:3,eagle3:4,draft_sd:5" \\
  --target-model "Qwen/Qwen3-8B" \\
  --draft-model "Qwen/Qwen2.5-1.5B-Instruct" \\
  --eagle3-model "RedHatAI/Qwen3-8B-speculator.eagle3" \\
  --input-jsonl \\
    {input_lines} \\
  --gpu-memory-utilization 0.60 \\
  2>&1 | tee "$OUT_DIR/run.log"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="v1", help="Version tag for reproducible output folder.")
    args = ap.parse_args()

    run_tag = args.run_tag

    result_root = Path(f"results/apexp_block/config_large_500/evaluation/table3_apex_ablation_cap50_explicit_{run_tag}")
    run_root = result_root / "apex_hot_runs"
    log_root = Path(f"slurm_logs/table3_apex_ablation_cap50_explicit/{run_tag}")
    variant_dir = log_root / "variant_scripts"

    result_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    variant_dir.mkdir(parents=True, exist_ok=True)

    for w in WORKLOADS:
        p = Path(w["input_jsonl"])
        if not p.exists():
            raise SystemExit(f"Missing workload file: {p}")

    manifest = []
    scripts = []

    for anchor, cfg in ANCHORS.items():
        for abl in ABLATIONS:
            model_dir = cfg["base_model_dir"] if abl["model"] == "base" else cfg["no_survival_model_dir"]

            run_name = f"{anchor}__quality_balanced__k1_to_16__{abl['name']}"
            run_dir = run_root / run_name

            sp = variant_dir / f"{run_name}.sbatch"
            sp.write_text(
                sbatch_text(
                    run_name=run_name,
                    run_dir=run_dir,
                    model_dir=model_dir,
                    anchor=anchor,
                    paper_anchor=cfg["paper_name"],
                    ablation=abl,
                    log_dir=variant_dir,
                )
            )
            sp.chmod(0o755)
            scripts.append(sp)

            manifest.append({
                "run_tag": run_tag,
                "run_name": run_name,
                "anchor": anchor,
                "paper_anchor": cfg["paper_name"],
                "ablation_name": abl["name"],
                "paper_variant": abl["paper_variant"],
                "ablation_type": abl["type"],
                "profile": "quality_balanced",
                "candidate_set": "k1_to_16",
                "candidate_ks": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
                "model_dir": model_dir,
                "run_dir": str(run_dir),
                "script": str(sp),
                "workloads": WORKLOADS,
                "non_destructive": True,
                "overwrite_policy": "refuse_if_OUT_DIR_exists",
            })

    manifest_path = result_root / "table3_apex_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    script_list = " ".join(str(s) for s in scripts)
    master = log_root / "run_all_table3_ablation_serial.sbatch"

    master.write_text(f'''#!/bin/bash
#SBATCH --job-name=table3_apex_ablation_{run_tag}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=700G
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output={log_root / "run_all_table3_ablation_serial.out"}
#SBATCH --error={log_root / "run_all_table3_ablation_serial.err"}

set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-UNSET}}"
nvidia-smi || true

SCRIPTS=({script_list})

for s in "${{SCRIPTS[@]}}"; do
  echo
  echo "============================================================"
  echo "[START] $(date) $s"
  echo "============================================================"
  bash "$s"
  echo "============================================================"
  echo "[DONE]  $(date) $s"
  echo "============================================================"
done

echo "[ALL DONE] $(date)"
''')
    master.chmod(0o755)

    print("WROTE result root:", result_root)
    print("WROTE manifest:", manifest_path)
    print("WROTE variant scripts:", variant_dir)
    print("WROTE master:", master)
    print("N scripts:", len(scripts))
    print()
    print("Submit:")
    print(f"  sbatch {master}")


if __name__ == "__main__":
    main()
