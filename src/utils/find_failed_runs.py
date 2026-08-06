#!/usr/bin/env python
"""
find_failed_runs.py — locate runs that crashed and emit re-run commands
=======================================================================
A run directory is considered FAILED if it contains a config.json (the run
was launched) but no summary.csv (it never produced results). This catches
both the F_context_growth tokenizer crash and the draft_sd CUDA crashes.

Usage:
    python find_failed_runs.py --runs_dir results/Runs
    python find_failed_runs.py --runs_dir results/Runs --emit-commands \
        --runner src/phase2_workload_characterization/run_benchmark.py
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def find_failed(runs_dir: Path):
    failed = []
    for cfg in runs_dir.glob("**/config.json"):
        run_dir = cfg.parent
        if (run_dir / "summary.csv").exists():
            continue
        try:
            conf = json.loads(cfg.read_text())
        except Exception:
            conf = {}
        failed.append((run_dir, conf))
    return failed


def classify(conf: dict) -> str:
    """Best-effort root-cause label based on the run config."""
    method = conf.get("method", "")
    exp = str(conf.get("experiment", ""))
    if method == "draft_sd":
        return "draft_sd CUDA crash (engine init / dummy_run) — retry with enforce_eager"
    if exp.startswith("F_context"):
        return "F_context_growth (tokenizer ordering — fixed in current run_benchmark.py)"
    return "other — inspect log"


def to_command(conf: dict, runner: str, oneline: bool = False) -> str:
    """Reconstruct a single-run CLI from a saved config.json.

    If oneline=True, emit a single-line command with NO CUDA_VISIBLE_DEVICES
    prefix (the parallel dispatcher pins the GPU per worker).
    """
    if not conf:
        return "# (no config.json content — inspect manually)"
    conf = dict(conf)
    # Hardening for the draft_sd CUDA crash: force eager + extra memory headroom.
    extra_flags = []
    if conf.get("method") == "draft_sd":
        conf["gpu_memory_utilization"] = 0.75       # override in place (no duplicate)
        extra_flags.append("--enforce_eager")
    flag_order = [
        "workload", "run_name", "experiment", "input_jsonl", "limit",
        "max_prompt_tokens", "num_turns", "out_dir", "model", "draft_model",
        "eagle3_model", "method", "k", "ngram_lookup_min", "ngram_lookup_max",
        "max_tokens", "temperature", "top_p", "top_logprobs",
        "tensor_parallel_size", "gpu_memory_utilization", "seed",
    ]
    parts = [f"python {runner}"]
    for key in flag_order:
        if key in conf and conf[key] not in ("", None):
            val = json.dumps(conf[key]) if isinstance(conf[key], str) else conf[key]
            parts.append(f"--{key} {val}")
    parts += extra_flags
    if oneline:
        return " ".join(parts)
    return " \\\n  ".join(parts)


PARALLEL_HEADER = r'''#!/usr/bin/env bash
# Auto-generated: re-run failed jobs across NGPU GPUs, one task per GPU.
# Uses a flock work-queue so a freed GPU immediately pulls the next job
# (load-balanced — better than round-robin when job durations vary).
set -u
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # required: parent inits CUDA via set_seed

NGPU=${NGPU:-8}
CMDS_FILE="${CMDS_FILE:-commands.txt}"
LOGDIR="${LOGDIR:-rerun_logs}"
mkdir -p "$LOGDIR"

mapfile -t CMDS < "$CMDS_FILE"
echo 0 > .queue.idx
: > .queue.lock

run_worker () {
  local gpu=$1
  while :; do
    exec 9>>.queue.lock; flock 9
    local idx; idx=$(<.queue.idx)
    if [ "$idx" -ge "${#CMDS[@]}" ]; then flock -u 9; break; fi
    echo $((idx + 1)) > .queue.idx
    flock -u 9
    local cmd="${CMDS[$idx]}"
    [ -z "$cmd" ] && continue
    echo "[GPU $gpu] job $idx/${#CMDS[@]}"
    CUDA_VISIBLE_DEVICES=$gpu $cmd > "$LOGDIR/gpu${gpu}_job${idx}.log" 2>&1 \
      || echo "[GPU $gpu] job $idx FAILED (see $LOGDIR/gpu${gpu}_job${idx}.log)"
  done
}

for g in $(seq 0 $((NGPU - 1))); do run_worker "$g" & done
wait
echo "All workers done. Check $LOGDIR/ for any FAILED jobs, then re-aggregate."
'''



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/Runs")
    ap.add_argument("--emit-commands", action="store_true")
    ap.add_argument("--runner", default="src/phase2_workload_characterization/run_benchmark.py")
    ap.add_argument("--out", default=None, help="write sequential re-run commands to this .sh file")
    ap.add_argument("--gpus", type=int, default=0,
                    help="If >0, emit a GPU-parallel work-queue: commands.txt + "
                         "rerun_parallel.sh that runs one task per GPU across N GPUs.")
    args = ap.parse_args()

    failed = find_failed(Path(args.runs_dir))
    print(f"Found {len(failed)} failed run dir(s) (config.json present, summary.csv missing)\n")

    by_study = Counter()
    by_cause = Counter()
    for run_dir, conf in failed:
        study = str(conf.get("experiment") or run_dir.parts[-4])
        by_study[study] += 1
        by_cause[classify(conf)] += 1

    print("By study:")
    for s, n in sorted(by_study.items()):
        print(f"  {s:28s} {n:4d}")
    print("\nBy likely cause:")
    for c, n in by_cause.most_common():
        print(f"  {n:4d}  {c}")

    # ── GPU-parallel work-queue ───────────────────────────────────────────
    if args.gpus and args.gpus > 0:
        cmds = [to_command(conf, args.runner, oneline=True) for _, conf in failed]
        Path("commands.txt").write_text("\n".join(cmds) + "\n")
        header = PARALLEL_HEADER.replace("NGPU=${NGPU:-8}", f"NGPU=${{NGPU:-{args.gpus}}}")
        Path("rerun_parallel.sh").write_text(header)
        print(f"\nWrote {len(cmds)} commands -> commands.txt")
        print(f"Wrote dispatcher  -> rerun_parallel.sh  ({args.gpus} GPUs, work-queue)")
        print("Run with:  bash rerun_parallel.sh")
        return

    if args.emit_commands:
        lines = ["#!/usr/bin/env bash", "set -e",
                 "export VLLM_WORKER_MULTIPROC_METHOD=spawn", ""]
        for run_dir, conf in failed:
            lines.append(f"# {run_dir}")
            lines.append(to_command(conf, args.runner))
            lines.append("")
        text = "\n".join(lines)
        if args.out:
            Path(args.out).write_text(text)
            print(f"\nWrote sequential re-run script -> {args.out}")
        else:
            print("\n" + "=" * 60 + "\nRE-RUN COMMANDS\n" + "=" * 60)
            print(text[:2000])


if __name__ == "__main__":
    main()
