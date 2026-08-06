from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import pandas as pd


METHODS = ["ngram_sd", "eagle3", "draft_sd"]


def now() -> float:
    return time.time()


def sha1_text(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()[:16]


def infer_workload(path: str) -> str:
    name = Path(path).name
    if "mathematical_reasoning" in name:
        return "mathematical_reasoning"
    if "long_horizon_swe" in name:
        return "long_horizon_swe"
    if "long_context_completion" in name:
        return "long_context_completion"
    if "long_chain_reasoning" in name:
        return "long_chain_reasoning"
    if "conversational_generation_test_sft" in name:
        return "conversational_generation_sft"
    if "conversational_generation_test_gen" in name:
        return "conversational_generation_gen"
    if "code_gen" in name:
        return "code_gen"
    return Path(path).stem


def extract_prompt(obj: dict[str, Any]) -> str:
    for k in ["prompt", "prompt_text", "input", "question", "text"]:
        if k in obj and obj[k] is not None:
            return str(obj[k])

    if "messages" in obj and isinstance(obj["messages"], list):
        parts = []
        for m in obj["messages"]:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(m))
        return "\n".join(parts)

    return json.dumps(obj, sort_keys=True)


def workload_max_tokens(workload: str) -> int:
    if workload in {"long_chain_reasoning", "long_horizon_swe"}:
        return 1024
    if workload == "long_context_completion":
        return 2048
    return 512


def score_col(df: pd.DataFrame) -> str:
    candidates = [
        "pred_tps",
        "pred_actual_tps",
        "predicted_tps",
        "router_score",
        "score",
        "pred",
        "actual_tps",
        "tokens_per_sec",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"No score column found in {list(df.columns)}")


@dataclass
class Action:
    method: str
    k: int
    temperature: float = 0.0
    source: str = "unknown"


class SlowRouter:
    """
    Online slow-router wrapper.

    For prompts already present in test_candidates_scored.csv, this uses the
    trained router's scored table exactly. For a brand-new prompt, it falls back
    to best global fixed ngram k=16 unless the router.pkl path is later wired
    with a fully reliable arbitrary-prompt feature builder.
    """

    def __init__(self, slow_router_dir: str):
        self.slow_router_dir = Path(slow_router_dir)
        self.prompt_map: dict[str, Action] = {}
        self._load_scored_prompt_map()

    def _load_scored_prompt_map(self):
        p = self.slow_router_dir / "test_candidates_scored.csv"
        if not p.exists():
            print(f"[WARN] missing {p}; using fallback action only", file=sys.stderr)
            return

        df = pd.read_csv(p, keep_default_na=False)
        sc = score_col(df)

        text_col = None
        for c in ["prompt_text", "prompt", "input_text"]:
            if c in df.columns:
                text_col = c
                break

        if text_col is None:
            print(f"[WARN] no prompt text column in {p}; using fallback action only", file=sys.stderr)
            return

        k_col = "k_requested" if "k_requested" in df.columns else "k"
        idx = df.groupby(text_col)[sc].idxmax()
        best = df.loc[idx].copy()

        for _, r in best.iterrows():
            prompt = str(r[text_col])
            method = str(r["method"])
            k = int(float(r[k_col]))
            temp = float(r["temperature"]) if "temperature" in r and str(r["temperature"]) != "" else 0.0
            self.prompt_map[prompt] = Action(method=method, k=k, temperature=temp, source="slow_router_scored_table")

        print(f"[router] loaded exact prompt actions: {len(self.prompt_map)}")

    def choose(self, prompt: str, workload: str) -> Action:
        if prompt in self.prompt_map:
            return self.prompt_map[prompt]

        # Safe fallback for arbitrary prompt. Replace this with full router.pkl
        # inference once arbitrary-prompt feature construction is verified.
        return Action(method="ngram_sd", k=16, temperature=0.0, source="fallback_ngram_k16")


@dataclass
class Task:
    task_id: str
    pool: str
    workload: str
    source_file: str
    source_index: int
    global_index: int
    prompt_hash: str
    prompt: str
    method: str
    slow_k: int
    temperature: float
    max_tokens: int
    router_source: str


def write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def manifest_row(task: Task, run_dir: Path, input_jsonl: Path, engine_k: int, target_model: str,
                 draft_model: str, eagle3_model: str, seed: int) -> dict[str, Any]:
    method = task.method

    row = {
        "priority": 0,
        "job_id": task.task_id,
        "source_experiments": f"realtime_{task.pool}",
        "phase": 2,
        "workload": task.workload,
        "input_jsonl": str(input_jsonl),
        "method": method,
        "k": int(engine_k),
        "temperature": float(task.temperature),
        "ngram_lookup_min": "NA",
        "ngram_lookup_max": "NA",
        "draft_model": "NA",
        "eagle3_model": "NA",
        "target_model": target_model,
        "max_prompt_tokens": 0,
        "num_turns": 1,
        "limit": 1,
        "max_tokens": int(task.max_tokens),
        "seed": int(seed),
        "run_dir": str(run_dir),
    }

    if method == "ngram_sd":
        row["ngram_lookup_min"] = "1"
        row["ngram_lookup_max"] = "4"
    elif method == "draft_sd":
        row["draft_model"] = draft_model
    elif method == "eagle3":
        row["eagle3_model"] = eagle3_model
    else:
        raise RuntimeError(f"Unknown method: {method}")

    return row


def run_task(task: Task, args, gpu: str):
    task_root = Path(args.out_dir) / task.pool / task.method / task.task_id
    run_dir = task_root / "run"
    input_jsonl = task_root / "input.jsonl"
    manifest = task_root / "manifest.tsv"
    log_path = task_root / "run.log"
    result_path = Path(args.out_dir) / "results" / f"{task.pool}_{task.task_id}.json"

    task_root.mkdir(parents=True, exist_ok=True)

    input_obj = {
        "prompt": task.prompt,
        "prompt_text": task.prompt,
        "prompt_hash": task.prompt_hash,
        "prompt_id": task.prompt_hash,
        "workload": task.workload,
        "source_file": task.source_file,
        "source_index": task.source_index,
        "global_index": task.global_index,
    }
    input_jsonl.write_text(json.dumps(input_obj) + "\n")

    # slow-only: engine uses exactly slow_k.
    # slow+fast: engine launches at max_engine_k, controller starts at slow_k.
    engine_k = task.slow_k if task.pool == "slow_only" else int(args.max_engine_k)

    row = manifest_row(
        task=task,
        run_dir=run_dir,
        input_jsonl=input_jsonl,
        engine_k=engine_k,
        target_model=args.target_model,
        draft_model=args.draft_model,
        eagle3_model=args.eagle3_model,
        seed=args.seed,
    )

    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["N_GPUS"] = "1"
    env["FORCE"] = "1"
    env["OUT_ROOT"] = str(task_root / "runmeta")
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{Path.cwd()}:{env.get('PYTHONPATH','')}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if task.pool == "slow_fast":
        env["APEXP_ONLINE_FAST"] = "1"
        env["APEXP_BLOCK_TRACE"] = "1"
        env["APEXP_TRACE_STRICT"] = "0"
        env["APEXP_DEFAULT_K"] = str(args.max_engine_k)
        env["APEXP_INITIAL_K"] = str(task.slow_k)
        env["APEXP_MIN_K"] = str(args.fast_min_k)
        env["APEXP_MAX_K"] = str(args.max_engine_k)
        env["APEXP_FAST_CANDIDATE_KS"] = args.fast_candidate_ks
        env["APEXP_FAST_WINDOW"] = str(args.fast_window)
        env["APEXP_FAST_MIN_OBS"] = str(args.fast_min_obs)
        env["APEXP_FAST_DOWN_RATIO"] = str(args.fast_down_ratio)
        env["APEXP_FAST_UP_RATIO"] = str(args.fast_up_ratio)
        env["APEXP_FAST_UP_FULL_RATE"] = str(args.fast_up_full_rate)
        env["APEXP_ONLINE_FAST_TRACE"] = str(Path(args.out_dir) / "online_fast_traces" / f"{task.method}_gpu{gpu}.jsonl")
    else:
        env.pop("APEXP_ONLINE_FAST", None)
        env.pop("APEXP_FORCE_ACTIVE_K", None)

    t0 = now()
    with log_path.open("w") as log:
        proc = subprocess.run(
            ["bash", args.runner_script, str(manifest)],
            cwd=args.workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    t1 = now()

    result = {
        **asdict(task),
        "gpu": gpu,
        "engine_k": engine_k,
        "run_dir": str(run_dir),
        "manifest": str(manifest),
        "log_path": str(log_path),
        "returncode": proc.returncode,
        "wall_s": t1 - t0,
        "status": "SUCCESS" if proc.returncode == 0 else "FAILED",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))


def worker_loop(pool: str, method: str, gpu: str, q: Queue, args):
    print(f"[worker start] pool={pool} method={method} gpu={gpu}", flush=True)
    while True:
        task = q.get()
        if task is None:
            print(f"[worker stop] pool={pool} method={method} gpu={gpu}", flush=True)
            break
        try:
            run_task(task, args, gpu)
        except Exception as e:
            err_path = Path(args.out_dir) / "errors" / f"{pool}_{method}_{task.task_id}.json"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(json.dumps({
                "task": asdict(task),
                "error": repr(e),
            }, indent=2, sort_keys=True))


def parse_gpu_map(x: str) -> dict[str, str]:
    out = {}
    for part in x.split(","):
        k, v = part.split(":")
        out[k.strip()] = v.strip()
    for m in METHODS:
        if m not in out:
            raise RuntimeError(f"Missing GPU for method {m} in map {x}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow-router-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--input-jsonl", nargs="+", required=True)
    ap.add_argument("--max-prompts-per-file", type=int, default=0)

    ap.add_argument("--workdir", default="/fsx/jmanvi/Internship_project/ASD")
    ap.add_argument("--runner-script", default="scripts/apexp/run_config_manifest_8gpu.sh")

    ap.add_argument("--slow-gpus", default="ngram_sd:0,eagle3:1,draft_sd:2")
    ap.add_argument("--fast-gpus", default="ngram_sd:3,eagle3:4,draft_sd:5")

    ap.add_argument("--target-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eagle3-model", default="RedHatAI/Qwen3-8B-speculator.eagle3")
    ap.add_argument("--max-engine-k", type=int, default=16)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast-min-k", type=int, default=1)
    ap.add_argument("--fast-candidate-ks", default="1,2,4,8,16")
    ap.add_argument("--fast-window", type=int, default=4)
    ap.add_argument("--fast-min-obs", type=int, default=4)
    ap.add_argument("--fast-down-ratio", type=float, default=0.45)
    ap.add_argument("--fast-up-ratio", type=float, default=0.85)
    ap.add_argument("--fast-up-full-rate", type=float, default=0.75)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    router = SlowRouter(args.slow_router_dir)
    slow_gpus = parse_gpu_map(args.slow_gpus)
    fast_gpus = parse_gpu_map(args.fast_gpus)

    queues: dict[tuple[str, str], Queue] = {}
    procs: list[Process] = []

    for pool, gpu_map in [("slow_only", slow_gpus), ("slow_fast", fast_gpus)]:
        for method in METHODS:
            q = Queue()
            queues[(pool, method)] = q
            p = Process(target=worker_loop, args=(pool, method, gpu_map[method], q, args))
            p.start()
            procs.append(p)

    dispatch_log = out_dir / "dispatch_log.jsonl"
    global_i = 0

    for input_path in args.input_jsonl:
        workload = infer_workload(input_path)
        count = 0

        with open(input_path) as f:
            for local_i, line in enumerate(f):
                if args.max_prompts_per_file and count >= args.max_prompts_per_file:
                    break

                obj = json.loads(line)
                prompt = extract_prompt(obj)
                ph = sha1_text(prompt)
                action = router.choose(prompt, workload)

                base = {
                    "event": "slow_router_decision",
                    "time": now(),
                    "global_index": global_i,
                    "source_file": input_path,
                    "source_index": local_i,
                    "workload": workload,
                    "prompt_hash": ph,
                    "method": action.method,
                    "slow_k": action.k,
                    "temperature": action.temperature,
                    "router_source": action.source,
                }
                write_jsonl(dispatch_log, base)

                for pool in ["slow_only", "slow_fast"]:
                    task = Task(
                        task_id=f"p{global_i:06d}_{workload}_{ph}",
                        pool=pool,
                        workload=workload,
                        source_file=input_path,
                        source_index=local_i,
                        global_index=global_i,
                        prompt_hash=ph,
                        prompt=prompt,
                        method=action.method,
                        slow_k=action.k,
                        temperature=action.temperature,
                        max_tokens=workload_max_tokens(workload),
                        router_source=action.source,
                    )
                    queues[(pool, action.method)].put(task)

                global_i += 1
                count += 1

    for q in queues.values():
        q.put(None)

    for p in procs:
        p.join()

    print(f"[done] dispatched prompts: {global_i}")
    print(f"[done] out_dir: {out_dir}")


if __name__ == "__main__":
    main()
