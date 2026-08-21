#!/usr/bin/env python3
from pathlib import Path
import json
import re

PARTITION = "ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40"

LARGE_ROOT = Path("results/apexp_block/config_large_500")
JOB_DIR = Path("slurm_logs/paper_eval_cap50_profile_kgrid/apex")
EVAL_ROOT = LARGE_ROOT / "evaluation/paper_cap50_profile_kgrid"
RUN_ROOT = EVAL_ROOT / "apex_hot_runs"
SLOW_ROUTER = LARGE_ROOT / "week2/request_prompt_router_d128"

JOB_DIR.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)

models = [
    LARGE_ROOT / "week2/survival_utility_controller_v4/speed_first",
    LARGE_ROOT / "week2/survival_utility_controller_v4/balanced",
    LARGE_ROOT / "week2/survival_utility_controller_v4/efficient",
    LARGE_ROOT / "week2/survival_utility_controller_v4b_rankgroup/speed_lr5e4_rank2",
    LARGE_ROOT / "week2/survival_utility_controller_v4b_rankgroup/speed_lr3e4_rank3",
    LARGE_ROOT / "week2/survival_utility_controller_v4b_rankgroup/balanced_lr5e4_rank2",
    LARGE_ROOT / "week2/survival_utility_controller_v4b_rankgroup/efficient_lr5e4_rank2",
]

workload_files = [
    "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/code_gen_cap50.jsonl",
    "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl",
    "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_context_completion_cap50.jsonl",
    "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_chain_reasoning_cap50.jsonl",
    "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_horizon_swe_cap50.jsonl",
    "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/hardware_gen_cap50.jsonl",
    "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_sft_cap50.jsonl",
    "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_gen_cap50.jsonl",
]

candidate_sets = {
    "k1_to_16": {
        "candidate_csv": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
        "min_k": 1,
    },
    "k3_to_16": {
        "candidate_csv": "3,4,5,6,7,8,9,10,11,12,13,14,15,16",
        "min_k": 3,
    },
    "k2_4_8_12_16": {
        "candidate_csv": "2,4,8,12,16",
        "min_k": 2,
    },
}

profiles = {
    # Profile 1: strict TPS preservation while still caring about acceptance/waste.
    "tps_strict": {
        "env": {
            "APEXP_LEARNED_FAST_ASYNC": "1",
            "APEXP_LEARNED_FAST_EVERY_N": "4",
            "APEXP_LEARNED_FAST_FIRST_WINDOW": "3",
            "APEXP_FAST_WINDOW": "3",
            "APEXP_FAST_MIN_OBS": "3",

            "APEXP_ONLINE_SCORE_UTILITY_WEIGHT": "0.8",
            "APEXP_ONLINE_SCORE_ACCEPT_WEIGHT": "0.12",
            "APEXP_ONLINE_SCORE_ACCEPT_RATE_WEIGHT": "0.25",
            "APEXP_ONLINE_SCORE_TPS_WEIGHT": "0.90",
            "APEXP_ONLINE_SCORE_WASTE_WEIGHT": "0.25",
            "APEXP_ONLINE_SCORE_BLOCK_PENALTY_WEIGHT": "0.45",
            "APEXP_ONLINE_MIN_EXPECTED_ACCEPTED": "1.25",
            "APEXP_ONLINE_MIN_ACCEPT_RATE": "0.05",
            "APEXP_ONLINE_BAD_ACTION_PENALTY": "4.0",
        },
    },

    # Profile 2: paper-friendly middle point.
    "quality_balanced": {
        "env": {
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
        },
    },

    # Profile 3: strict acceptance/waste. TPS may drop.
    "quality_strict": {
        "env": {
            "APEXP_LEARNED_FAST_ASYNC": "1",
            "APEXP_LEARNED_FAST_EVERY_N": "4",
            "APEXP_LEARNED_FAST_FIRST_WINDOW": "3",
            "APEXP_FAST_WINDOW": "3",
            "APEXP_FAST_MIN_OBS": "3",

            "APEXP_ONLINE_SCORE_UTILITY_WEIGHT": "0.4",
            "APEXP_ONLINE_SCORE_ACCEPT_WEIGHT": "0.25",
            "APEXP_ONLINE_SCORE_ACCEPT_RATE_WEIGHT": "1.35",
            "APEXP_ONLINE_SCORE_TPS_WEIGHT": "0.02",
            "APEXP_ONLINE_SCORE_WASTE_WEIGHT": "1.60",
            "APEXP_ONLINE_SCORE_BLOCK_PENALTY_WEIGHT": "0.02",
            "APEXP_ONLINE_MIN_EXPECTED_ACCEPTED": "0.35",
            "APEXP_ONLINE_MIN_ACCEPT_RATE": "0.20",
            "APEXP_ONLINE_BAD_ACTION_PENALTY": "3.0",
        },
    },
}

def slug(x: str) -> str:
    x = str(x).replace("/", "_")
    x = re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")

def model_label(path: Path) -> str:
    return slug(f"{path.parent.name}_{path.name}")

missing = [x for x in workload_files if not Path(x).exists()]
if missing:
    raise SystemExit("Missing workload files:\n" + "\n".join(missing))

missing_models = [str(x) for x in models if not x.exists()]
if missing_models:
    raise SystemExit("Missing model dirs:\n" + "\n".join(missing_models))

written = []
manifest = []

for model in models:
    mlabel = model_label(model)

    for profile_name, profile_cfg in profiles.items():
        for cset_name, cset_cfg in candidate_sets.items():
            candidate_csv = cset_cfg["candidate_csv"]
            min_k = int(cset_cfg["min_k"])

            run_name = f"{mlabel}__{profile_name}__{cset_name}"
            job_name = slug(f"apex_{profile_name}_{cset_name}_{mlabel}")[:60]
            out_dir = RUN_ROOT / run_name
            script = JOB_DIR / f"{run_name}.sbatch"

            input_lines = " \\\n    ".join(workload_files)
            env_lines = "\n".join(
                f"export {k}={v}" for k, v in profile_cfg["env"].items()
            )

            text = f"""#!/bin/bash
#SBATCH --job-name={job_name[:48]}
#SBATCH --output={JOB_DIR}/{job_name}_%j.out
#SBATCH --error={JOB_DIR}/{job_name}_%j.err
#SBATCH --partition={PARTITION}
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=420G

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

export APEXP_ONLINE_FAST=1
export APEXP_LEARNED_FAST_MODEL_DIR="{model}"
export APEXP_LEARNED_FAST_DEVICE=cpu
export APEXP_LEARNED_FAST_TRACE_SCORES=0

export APEXP_FAST_CANDIDATE_KS="{candidate_csv}"
export APEXP_MIN_K={min_k}
export APEXP_MAX_K=16

{env_lines}

export OUT_DIR="{out_dir}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "[APEX PROFILE-KGRID JOB]"
echo "run_name={run_name}"
echo "model_variant={mlabel}"
echo "profile={profile_name}"
echo "candidate_set={cset_name}"
echo "candidate_ks={candidate_csv}"
echo "min_k={min_k}"
echo "out_dir=$OUT_DIR"
which nvcc || true
nvcc --version || true

python3 src/apexp/week2/realtime_hot_e2e.py \\
  --slow-router-dir "{SLOW_ROUTER}" \\
  --out-dir "$OUT_DIR" \\
  --max-prompts-per-file 0 \\
  --fast-min-k {min_k} \\
  --fast-candidate-ks "{candidate_csv}" \\
  --fast-window 3 \\
  --fast-min-obs 3 \\
  --slow-gpus "ngram_sd:0,eagle3:1,draft_sd:2" \\
  --fast-gpus "ngram_sd:3,eagle3:4,draft_sd:5" \\
  --target-model "Qwen/Qwen3-8B" \\
  --draft-model "Qwen/Qwen2.5-1.5B-Instruct" \\
  --eagle3-model "RedHatAI/Qwen3-8B-speculator.eagle3" \\
  --input-jsonl \\
    {input_lines} \\
  --gpu-memory-utilization 0.50 \\
  2>&1 | tee "$OUT_DIR/run.log"
"""

            script.write_text(text)
            written.append(script)

            manifest.append({
                "run_name": run_name,
                "model_variant": mlabel,
                "model_dir": str(model),
                "profile": profile_name,
                "candidate_set": cset_name,
                "candidate_ks": candidate_csv,
                "min_k": min_k,
                **profile_cfg["env"],
            })

submit = JOB_DIR / "submit_all.sh"
submit.write_text(
    "#!/bin/bash\nset -euo pipefail\n"
    + "\n".join(f"sbatch {x}" for x in written)
    + "\n"
)
submit.chmod(0o755)

jobs_txt = JOB_DIR / "submit_all_jobs.txt"
jobs_txt.write_text("\n".join(str(x) for x in written) + "\n")

manifest_path = EVAL_ROOT / "profile_kgrid_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

print(f"WROTE {len(written)} jobs to {JOB_DIR}")
print(f"WROTE manifest to {manifest_path}")
