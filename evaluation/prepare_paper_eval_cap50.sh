#!/bin/bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD

PARTITION="ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40"
LARGE_ROOT="results/apexp_block/config_large_500"
EVAL_ROOT="$LARGE_ROOT/evaluation/paper_cap50"

APEX_JOB_DIR="slurm_logs/paper_eval_cap50/apex"
BASE_JOB_DIR="slurm_logs/paper_eval_cap50/baselines"
DATA_DIR="data/benchmarks/By_split_phase_1/apex_paper_eval_cap50"

APEX_RUN_ROOT="$EVAL_ROOT/apex_hot_runs"
BASE_RUN_ROOT="$EVAL_ROOT/fixed_baseline_runs"

mkdir -p "$APEX_JOB_DIR" "$BASE_JOB_DIR" "$DATA_DIR" "$EVAL_ROOT"

echo "============================================================"
echo "PATCHING RUNTIME SAFETY FIXES"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path

# ------------------------------------------------------------
# 1. Fix learned online controller
# ------------------------------------------------------------
p = Path("src/apexp/runtime/online_fast_controller.py")
txt = p.read_text()

txt = txt.replace(
    "from src.apexp.runtime.learned_fast_survival_policy_fast import FastLearnedSurvivalPolicy",
    "from apexp.runtime.learned_fast_survival_policy_fast import FastLearnedSurvivalPolicy",
)

needle = '        self.up_full_rate = float(os.environ.get("APEXP_FAST_UP_FULL_RATE", "0.75"))\n\n        self.control_file = os.environ.get("APEXP_REQUEST_CONTROL_FILE", "")'
replacement = '        self.up_full_rate = float(os.environ.get("APEXP_FAST_UP_FULL_RATE", "0.75"))\n        self.learned_trace_scores = os.environ.get("APEXP_LEARNED_FAST_TRACE_SCORES", "0") == "1"\n\n        self.control_file = os.environ.get("APEXP_REQUEST_CONTROL_FILE", "")'
if "self.learned_trace_scores =" not in txt:
    txt = txt.replace(needle, replacement)

p.write_text(txt)
print("patched", p)

# ------------------------------------------------------------
# 2. Fix learned fast policy imports
# ------------------------------------------------------------
p = Path("src/apexp/runtime/learned_fast_survival_policy_fast.py")
txt = p.read_text()
txt = txt.replace("from src.apexp.week2.neural_controller_lib import", "from apexp.week2.neural_controller_lib import")
txt = txt.replace("from src.apexp.week2.prompt_embedder import", "from apexp.week2.prompt_embedder import")
txt = txt.replace("from src.apexp.week2.train_survival_utility_controller_v4 import", "from apexp.week2.train_survival_utility_controller_v4 import")
p.write_text(txt)
print("patched", p)

# ------------------------------------------------------------
# 3. Fix realtime hot runner:
#    - hardware_gen label
#    - prompt extraction
#    - prevent slow_only from being clipped by fast candidate sets
# ------------------------------------------------------------
p = Path("src/apexp/week2/realtime_hot_e2e.py")
txt = p.read_text()

if 'if "hardware_gen" in name:' not in txt:
    txt = txt.replace(
        '    if "code_gen" in name:\n        return "code_gen"\n    return Path(path).stem',
        '    if "hardware_gen" in name:\n        return "hardware_gen"\n    if "code_gen" in name:\n        return "code_gen"\n    return Path(path).stem',
    )

txt = txt.replace(
    'for k in ["prompt", "prompt_text", "input", "question", "text"]:',
    'for k in ["prompt", "prompt_text", "input", "question", "problem_statement", "text", "content"]:',
)

txt = txt.replace(
    '"candidate_ks": args.fast_candidate_ks_list,',
    '"candidate_ks": (args.fast_candidate_ks_list if pool == "slow_fast" else [int(task.slow_k)]),',
)

p.write_text(txt)
print("patched", p)

# ------------------------------------------------------------
# 4. Make run_benchmark eagle3 config consistent with realtime runner.
# ------------------------------------------------------------
p = Path("src/layer0_data_collection/run_benchmark.py")
txt = p.read_text()
txt = txt.replace(
    'return {"method": "eagle3", "model": eagle3_model,\n                "num_speculative_tokens": k}',
    'return {"method": "eagle", "model": eagle3_model,\n                "num_speculative_tokens": k}',
)
p.write_text(txt)
print("patched", p)
PY

echo
echo "============================================================"
echo "PYTHON COMPILE CHECK"
echo "============================================================"

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python3 -m py_compile \
  src/apexp/week2/realtime_hot_e2e.py \
  src/apexp/runtime/online_fast_controller.py \
  src/apexp/runtime/learned_fast_survival_policy_fast.py \
  src/layer0_data_collection/run_benchmark.py

echo
echo "============================================================"
echo "CREATING EXACT CAP-50 EVALUATION FILES"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path
import json
import random
import hashlib

ROOT = Path("data/benchmarks/By_split_phase_1")
OUT = ROOT / "apex_paper_eval_cap50"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
CAP = 50

WORKLOADS = [
    {
        "workload": "code_gen",
        "files": [ROOT / "code_gen_test.jsonl"],
        "out": OUT / "code_gen_cap50.jsonl",
    },
    {
        "workload": "mathematical_reasoning",
        "files": [ROOT / "mathematical_reasoning_test.jsonl"],
        "out": OUT / "mathematical_reasoning_cap50.jsonl",
    },
    {
        "workload": "long_context_completion",
        "files": [
            ROOT / "long_context_completion_synthetic.jsonl",
            ROOT / "long_context_completion.jsonl",
        ],
        "out": OUT / "long_context_completion_cap50.jsonl",
    },
    {
        "workload": "long_chain_reasoning",
        "files": [
            ROOT / "long_chain_reasoning_synthetic.jsonl",
            ROOT / "long_chain_reasoning.jsonl",
        ],
        "out": OUT / "long_chain_reasoning_cap50.jsonl",
    },
    {
        "workload": "long_horizon_swe",
        "files": [ROOT / "long_horizon_swe_test.jsonl"],
        "out": OUT / "long_horizon_swe_cap50.jsonl",
    },
    {
        "workload": "hardware_gen",
        "files": [ROOT / "hardware_gen_test.jsonl"],
        "out": OUT / "hardware_gen_cap50.jsonl",
    },
    {
        "workload": "conversational_generation_sft",
        "files": [ROOT / "conversational_generation_test_sft.jsonl"],
        "out": OUT / "conversational_generation_sft_cap50.jsonl",
    },
    {
        "workload": "conversational_generation_gen",
        "files": [ROOT / "conversational_generation_test_gen.jsonl"],
        "out": OUT / "conversational_generation_gen_cap50.jsonl",
    },
]

def extract_prompt(obj):
    for k in ["prompt", "prompt_text", "input", "question", "problem_statement", "text", "content"]:
        if k in obj and obj[k] is not None:
            return str(obj[k])
    if "messages" in obj and isinstance(obj["messages"], list):
        return "\n".join(
            f"{m.get('role','')}: {m.get('content','')}" if isinstance(m, dict) else str(m)
            for m in obj["messages"]
        )
    return json.dumps(obj, sort_keys=True)

def prompt_hash(obj):
    return hashlib.sha1(extract_prompt(obj).encode("utf-8", errors="ignore")).hexdigest()

def read_jsonl(path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

grand_total = 0
print(f"{'workload':38s} {'source_rows':>12s} {'written':>8s} output")
print("-" * 95)

for spec in WORKLOADS:
    rng = random.Random(SEED)
    all_rows = []
    seen = set()

    # For long-chain/context, synthetic comes first, then train sample fills the rest.
    for idx, src in enumerate(spec["files"]):
        rows = read_jsonl(src)

        if idx == 0 and "synthetic" in src.name:
            ordered = rows
        else:
            ordered = rows[:]
            rng.shuffle(ordered)

        for r in ordered:
            h = prompt_hash(r)
            if h in seen:
                continue
            seen.add(h)

            rr = dict(r)
            rr["_apex_eval_workload"] = spec["workload"]
            rr["_apex_eval_source_file"] = str(src)

            if "synthetic" in src.name:
                rr["_apex_eval_source"] = "synthetic_seed"
            elif src.name.endswith("_test.jsonl"):
                rr["_apex_eval_source"] = "heldout_test_sample"
            else:
                rr["_apex_eval_source"] = "train_sampled_expansion"

            all_rows.append(rr)
            if len(all_rows) >= CAP:
                break

        if len(all_rows) >= CAP:
            break

    if len(all_rows) < CAP:
        raise RuntimeError(f"{spec['workload']} only has {len(all_rows)} rows, expected {CAP}")

    with spec["out"].open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    grand_total += len(all_rows)
    source_total = sum(len(read_jsonl(x)) for x in spec["files"])
    print(f"{spec['workload']:38s} {source_total:12d} {len(all_rows):8d} {spec['out']}")

print("-" * 95)
print(f"{'TOTAL PROMPTS':38s} {'':12s} {grand_total:8d}")
print(f"{'APEX generations per run':38s} {'':12s} {2 * grand_total:8d}")
PY

echo
echo "============================================================"
echo "PRE-FLIGHT MODEL / ROUTER CHECK"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path

router = Path("results/apexp_block/config_large_500/week2/request_prompt_router_d128")
models = [
    "results/apexp_block/config_large_500/week2/survival_utility_controller_v4/speed_first",
    "results/apexp_block/config_large_500/week2/survival_utility_controller_v4/balanced",
    "results/apexp_block/config_large_500/week2/survival_utility_controller_v4/efficient",
    "results/apexp_block/config_large_500/week2/survival_utility_controller_v4b_rankgroup/speed_lr5e4_rank2",
    "results/apexp_block/config_large_500/week2/survival_utility_controller_v4b_rankgroup/speed_lr3e4_rank3",
    "results/apexp_block/config_large_500/week2/survival_utility_controller_v4b_rankgroup/balanced_lr5e4_rank2",
    "results/apexp_block/config_large_500/week2/survival_utility_controller_v4b_rankgroup/efficient_lr5e4_rank2",
]
files = sorted(Path("data/benchmarks/By_split_phase_1/apex_paper_eval_cap50").glob("*.jsonl"))

print("router:", router, "OK" if router.exists() else "MISSING")
print("router.pkl:", (router / "router.pkl").exists())
print("request_prompt_router.pkl:", (router / "request_prompt_router.pkl").exists())
print("prompt_embedder.pkl:", (router / "prompt_embedder.pkl").exists())

print("\nmodels:")
for m in models:
    p = Path(m)
    needed = ["model.pt", "metadata.json", "preprocessor.pkl", "prompt_embedder.pkl"]
    ok = p.exists() and all((p / x).exists() for x in needed)
    print(("OK      " if ok else "MISSING "), m)

print("\ncap50 files:")
total = 0
for f in files:
    n = sum(1 for _ in f.open())
    total += n
    print(f"{f.name:55s} {n:6d}")
print("TOTAL", total)
PY

echo
echo "============================================================"
echo "GENERATING APEX LEARNED-CONTROLLER SLURM JOBS"
echo "============================================================"

rm -f "$APEX_JOB_DIR"/*.sbatch "$APEX_JOB_DIR"/submit_all.sh "$APEX_JOB_DIR"/submit_all_jobs.txt

python3 - <<'PY'
from pathlib import Path
import re

PARTITION = "ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40"
LARGE_ROOT = Path("results/apexp_block/config_large_500")
JOB_DIR = Path("slurm_logs/paper_eval_cap50/apex")
RUN_ROOT = LARGE_ROOT / "evaluation/paper_cap50/apex_hot_runs"
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

candidate_sets = [
    ("k1_2_4_8_16", "1,2,4,8,16"),
    ("k1_2_4_6_8_10_12_14_16", "1,2,4,6,8,10,12,14,16"),
    ("k1_to_16", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"),
    ("k2_4_6_8", "2,4,6,8"),
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

def slug(x):
    x = str(x).replace("/", "_")
    x = re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")

def model_label(p):
    p = Path(p)
    return slug(f"{p.parent.name}_{p.name}")

written = []

for model in models:
    label = model_label(model)
    for cname, csv in candidate_sets:
        vals = [int(x) for x in csv.split(",")]
        min_k = min(vals)
        run_name = f"{label}__{cname}"
        job_name = f"apex_{run_name}"[:60]
        out_dir = RUN_ROOT / run_name
        script = JOB_DIR / f"{run_name}.sbatch"

        input_lines = " \\\n    ".join(workload_files)

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
export APEXP_LEARNED_FAST_EVERY_N=1
export APEXP_LEARNED_FAST_SWITCH_MARGIN=0.03
export APEXP_LEARNED_FAST_TRACE_SCORES=0

export APEXP_ONLINE_FAST_TRACE=1
export APEXP_FAST_CANDIDATE_KS="{csv}"
export APEXP_MIN_K={min_k}
export APEXP_MAX_K=16
export APEXP_FAST_WINDOW=2
export APEXP_FAST_MIN_OBS=2

export OUT_DIR="{out_dir}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "[APEX JOB] run_name={run_name}"
echo "[APEX JOB] model={model}"
echo "[APEX JOB] candidate_ks={csv}"
echo "[APEX JOB] out_dir=$OUT_DIR"
which nvcc || true
nvcc --version || true

python3 src/apexp/week2/realtime_hot_e2e.py \\
  --slow-router-dir "{SLOW_ROUTER}" \\
  --out-dir "$OUT_DIR" \\
  --max-prompts-per-file 0 \\
  --fast-min-k {min_k} \\
  --fast-candidate-ks "{csv}" \\
  --fast-window 2 \\
  --fast-min-obs 2 \\
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

submit = JOB_DIR / "submit_all.sh"
submit.write_text("#!/bin/bash\nset -euo pipefail\n" + "\n".join(f"sbatch {p}" for p in written) + "\n")
submit.chmod(0o755)

jobs_txt = JOB_DIR / "submit_all_jobs.txt"
jobs_txt.write_text("\n".join(str(p) for p in written) + "\n")

print(f"WROTE {len(written)} APEX sbatch files to {JOB_DIR}")
PY

echo
echo "============================================================"
echo "GENERATING FIXED BASELINE SLURM JOBS"
echo "============================================================"

rm -f "$BASE_JOB_DIR"/*.sbatch "$BASE_JOB_DIR"/submit_all.sh "$BASE_JOB_DIR"/submit_all_jobs.txt

python3 - <<'PY'
from pathlib import Path

PARTITION = "ml-p5en-48xlarge-us-west-2d-tp-p5en-agentcore-eval40"
JOB_DIR = Path("slurm_logs/paper_eval_cap50/baselines")
RUN_ROOT = Path("results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs")

JOB_DIR.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)

methods = ["eagle3", "ngram_sd", "draft_sd"]
ks = [1, 2, 4, 8, 16]

workloads = [
    ("code_gen", "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/code_gen_cap50.jsonl", 512),
    ("mathematical_reasoning", "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl", 512),
    ("long_context_completion", "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_context_completion_cap50.jsonl", 2048),
    ("long_chain_reasoning", "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_chain_reasoning_cap50.jsonl", 1024),
    ("long_horizon_swe", "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_horizon_swe_cap50.jsonl", 1024),
    ("hardware_gen", "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/hardware_gen_cap50.jsonl", 512),
    ("conversational_generation_sft", "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_sft_cap50.jsonl", 512),
    ("conversational_generation_gen", "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_gen_cap50.jsonl", 512),
]

written = []

for method in methods:
    for k in ks:
        job_name = f"base_{method}_k{k}"
        script = JOB_DIR / f"{job_name}.sbatch"

        workload_cmds = []
        for workload, file, max_tokens in workloads:
            eager = "--enforce_eager" if method == "draft_sd" else ""
            workload_cmds.append(f"""
echo "[BASELINE] method={method} k={k} workload={workload}"

python3 src/layer0_data_collection/run_benchmark.py \\
  --workload "{workload}" \\
  --run_name "{workload}" \\
  --experiment "fixed_baseline_cap50" \\
  --input_jsonl "{file}" \\
  --limit 0 \\
  --out_dir "{RUN_ROOT}" \\
  --model "Qwen/Qwen3-8B" \\
  --draft_model "Qwen/Qwen2.5-1.5B-Instruct" \\
  --eagle3_model "RedHatAI/Qwen3-8B-speculator.eagle3" \\
  --method "{method}" \\
  --k {k} \\
  --max_tokens {max_tokens} \\
  --temperature 0.0 \\
  --gpu_memory_utilization 0.50 \\
  {eager}
""")

        text = f"""#!/bin/bash
#SBATCH --job-name=apex_{job_name}
#SBATCH --output={JOB_DIR}/apex_{job_name}_%j.out
#SBATCH --error={JOB_DIR}/apex_{job_name}_%j.err
#SBATCH --partition={PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

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

echo "[BASELINE JOB] method={method} k={k}"
echo "[BASELINE JOB] out_root={RUN_ROOT}"
which nvcc || true
nvcc --version || true

{''.join(workload_cmds)}
"""
        script.write_text(text)
        written.append(script)

submit = JOB_DIR / "submit_all.sh"
submit.write_text("#!/bin/bash\nset -euo pipefail\n" + "\n".join(f"sbatch {p}" for p in written) + "\n")
submit.chmod(0o755)

jobs_txt = JOB_DIR / "submit_all_jobs.txt"
jobs_txt.write_text("\n".join(str(p) for p in written) + "\n")

print(f"WROTE {len(written)} baseline sbatch files to {JOB_DIR}")
PY

echo
echo "============================================================"
echo "WRITING THROTTLED SUBMITTERS"
echo "============================================================"

cat > slurm_logs/paper_eval_cap50/submit_throttled.sh <<'BASH2'
#!/bin/bash
set -uo pipefail

JOB_LIST="${JOB_LIST:?Set JOB_LIST to submit_all_jobs.txt}"
MAX_ACTIVE="${MAX_ACTIVE:-1}"
POLL_SEC="${POLL_SEC:-60}"
SUBMIT_GAP_SEC="${SUBMIT_GAP_SEC:-30}"

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
BASH2

chmod +x slurm_logs/paper_eval_cap50/submit_throttled.sh

cat > "$APEX_JOB_DIR/submit_all_throttled.sh" <<'BASH2'
#!/bin/bash
set -euo pipefail
JOB_LIST="slurm_logs/paper_eval_cap50/apex/submit_all_jobs.txt" \
MAX_ACTIVE="${MAX_ACTIVE:-1}" \
POLL_SEC="${POLL_SEC:-60}" \
SUBMIT_GAP_SEC="${SUBMIT_GAP_SEC:-60}" \
bash slurm_logs/paper_eval_cap50/submit_throttled.sh
BASH2
chmod +x "$APEX_JOB_DIR/submit_all_throttled.sh"

cat > "$BASE_JOB_DIR/submit_all_throttled.sh" <<'BASH2'
#!/bin/bash
set -euo pipefail
JOB_LIST="slurm_logs/paper_eval_cap50/baselines/submit_all_jobs.txt" \
MAX_ACTIVE="${MAX_ACTIVE:-6}" \
POLL_SEC="${POLL_SEC:-60}" \
SUBMIT_GAP_SEC="${SUBMIT_GAP_SEC:-30}" \
bash slurm_logs/paper_eval_cap50/submit_throttled.sh
BASH2
chmod +x "$BASE_JOB_DIR/submit_all_throttled.sh"

echo
echo "============================================================"
echo "WRITING BASELINE SUMMARIZER"
echo "============================================================"

cat > evaluation/summarize_fixed_baseline_cap50.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_all_summary_csv(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("summary.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
            continue
        if df.empty:
            continue
        df["summary_path"] = str(p)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "k", "tokens_per_sec", "latency_s", "n_output_tokens",
        "acceptance_rate", "draft_tokens", "accepted_tokens_total",
        "rejected_tokens", "rejection_rate", "accepted_tokens_per_verifier_pass",
        "mean_entropy", "prompt_token_len",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = add_numeric(df)

    for c in ["draft_tokens", "accepted_tokens_total", "rejected_tokens"]:
        if c not in d.columns:
            d[c] = np.nan

    out = (
        d.groupby(group_cols, dropna=False)
        .agg(
            n=("tokens_per_sec", "size"),
            mean_tps=("tokens_per_sec", "mean"),
            median_tps=("tokens_per_sec", "median"),
            mean_latency_s=("latency_s", "mean"),
            median_latency_s=("latency_s", "median"),
            mean_output_tokens=("n_output_tokens", "mean"),
            mean_acceptance_rate=("acceptance_rate", "mean"),
            mean_rejection_rate=("rejection_rate", "mean"),
            mean_entropy=("mean_entropy", "mean"),
            total_draft_tokens=("draft_tokens", "sum"),
            total_accepted_tokens=("accepted_tokens_total", "sum"),
            total_rejected_tokens=("rejected_tokens", "sum"),
        )
        .reset_index()
    )

    out["wasted_token_rate"] = out["total_rejected_tokens"] / out["total_draft_tokens"].replace(0, np.nan)
    out["accepted_token_rate"] = out["total_accepted_tokens"] / out["total_draft_tokens"].replace(0, np.nan)
    out["wasted_token_pct"] = 100.0 * out["wasted_token_rate"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_all_summary_csv(root)
    if rows.empty:
        raise SystemExit(f"No summary.csv files found under {root}")

    rows = add_numeric(rows)
    rows.to_csv(out_dir / "fixed_baseline_request_rows.csv", index=False)

    by_workload = summarize_group(rows, ["method", "k", "workload"])
    by_workload.to_csv(out_dir / "fixed_baseline_by_workload_micro.csv", index=False)

    overall_micro = summarize_group(rows, ["method", "k"])
    overall_micro.to_csv(out_dir / "fixed_baseline_overall_micro.csv", index=False)

    macro = (
        by_workload.groupby(["method", "k"], dropna=False)
        .agg(
            n_workloads=("workload", "nunique"),
            macro_mean_tps=("mean_tps", "mean"),
            macro_median_tps=("median_tps", "mean"),
            macro_mean_latency_s=("mean_latency_s", "mean"),
            macro_acceptance_rate=("mean_acceptance_rate", "mean"),
            macro_rejection_rate=("mean_rejection_rate", "mean"),
            macro_wasted_token_rate=("wasted_token_rate", "mean"),
            macro_wasted_token_pct=("wasted_token_pct", "mean"),
        )
        .reset_index()
        .sort_values("macro_mean_tps", ascending=False)
    )
    macro.to_csv(out_dir / "fixed_baseline_overall_macro_by_workload.csv", index=False)

    print("WROTE:", out_dir)
    print("request rows:", len(rows))
    print("by workload rows:", len(by_workload))
    print("overall rows:", len(overall_micro))


if __name__ == "__main__":
    main()
PY

chmod +x evaluation/summarize_fixed_baseline_cap50.py

echo
echo "============================================================"
echo "WRITING SUMMARY SCRIPT"
echo "============================================================"

cat > slurm_logs/paper_eval_cap50/summarize_all.sh <<'BASH2'
#!/bin/bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python3 evaluation/03_summarize_hot_runs.py \
  --runs-root results/apexp_block/config_large_500/evaluation/paper_cap50/apex_hot_runs \
  --out-dir results/apexp_block/config_large_500/evaluation/paper_cap50/apex_hot_summary \
  --force

python3 evaluation/summarize_fixed_baseline_cap50.py \
  --root results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_runs \
  --out-dir results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_summary

echo
echo "APEX summary:"
echo "  results/apexp_block/config_large_500/evaluation/paper_cap50/apex_hot_summary"
echo
echo "Baseline summary:"
echo "  results/apexp_block/config_large_500/evaluation/paper_cap50/fixed_baseline_summary"
BASH2

chmod +x slurm_logs/paper_eval_cap50/summarize_all.sh

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "APEX jobs:      $APEX_JOB_DIR"
echo "Baseline jobs:  $BASE_JOB_DIR"
echo "Eval root:      $EVAL_ROOT"
echo
echo "APEX job count:"
ls "$APEX_JOB_DIR"/*.sbatch | wc -l
echo "Baseline job count:"
ls "$BASE_JOB_DIR"/*.sbatch | wc -l
