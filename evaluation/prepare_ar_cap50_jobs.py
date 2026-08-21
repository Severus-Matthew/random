#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path("results/apexp_block/config_large_500/evaluation/paper_cap50_profile_kgrid")
DATA_DIR = Path("data/benchmarks/By_split_phase_1/apex_paper_eval_cap50")
JOB_DIR = Path("slurm_logs/paper_eval_cap50_profile_kgrid/ar")
RUN_ROOT = ROOT / "fixed_runs"
PARTITION = "ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40"

TARGET_MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 512
LIMIT = 0
TEMP = 0.0
SEED = 42
GPU_MEM = 0.80


def infer_workload(p: Path) -> str:
    name = p.stem
    name = re.sub(r"_cap50$", "", name)
    name = re.sub(r"_test$", "", name)
    name = re.sub(r"_synthetic$", "", name)
    return name


def main():
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No jsonl files found in {DATA_DIR}")

    job_paths = []

    for inp in files:
        workload = infer_workload(inp)
        out_dir = RUN_ROOT / workload / "ar" / "ar_temp0.0"
        job_name = f"ar_{workload}"[:120]
        sbatch = JOB_DIR / f"{job_name}.sbatch"
        stdout = JOB_DIR / f"{job_name}.out"
        stderr = JOB_DIR / f"{job_name}.err"

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output={stdout}
#SBATCH --error={stderr}

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

# Important: do NOT overwrite CUDA_VISIBLE_DEVICES here.
# Slurm should decide which physical GPU this one-GPU AR job receives.

mkdir -p "{out_dir}"

echo "[AR CAP50 JOB]"
echo "workload={workload}"
echo "input_jsonl={inp}"
echo "out_dir={out_dir}"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-UNSET}}"
nvidia-smi

python3 src/layer0_data_collection/run_benchmark.py \\
  --workload "{workload}" \\
  --input_jsonl "{inp}" \\
  --out_dir "{out_dir}" \\
  --experiment "paper_cap50_ar" \\
  --run_name "{workload}_full" \\
  --method ar \\
  --k 1 \\
  --limit {LIMIT} \\
  --temperature {TEMP} \\
  --max_tokens {MAX_TOKENS} \\
  --seed {SEED} \\
  --enforce_eager \\
  --gpu_memory_utilization {GPU_MEM} \\
  --model "{TARGET_MODEL}"

echo "[done] {workload}"
"""
        sbatch.write_text(script)
        job_paths.append(sbatch)

    submit_all = JOB_DIR / "submit_all_ar.sh"
    submit_all.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "cd /fsx/jmanvi/Internship_project/ASD\n"
        f"for j in {JOB_DIR}/*.sbatch; do\n"
        "  echo \"Submitting $j\"\n"
        "  sbatch \"$j\"\n"
        "  sleep 15\n"
        "done\n"
    )
    submit_all.chmod(0o755)

    print(f"wrote {len(job_paths)} AR sbatch files in {JOB_DIR}")
    print(f"submit with: bash {submit_all}")


if __name__ == "__main__":
    main()
