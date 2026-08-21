#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from multiprocessing import Process
from pathlib import Path


ROOT = Path("/fsx/jmanvi/Internship_project/ASD")

FIXED_RUNS_ROOT = ROOT / "results/apexp_block/config_large_500/evaluation/final_evaluations/fixed_runs"
TMP_ROOT = ROOT / "results/apexp_block/config_large_500/evaluation/final_evaluations/eagle3_recommended_rerun_tmp_7gpu"
LOG_ROOT = ROOT / "slurm_logs/rerun_eagle3_recommended_7gpu"

KS = [1, 2, 4, 8, 16]

WORKLOADS = [
    {
        "workload": "code_gen",
        "input_jsonl": str(ROOT / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/code_gen_cap50.jsonl"),
        "max_tokens": 512,
    },
    {
        "workload": "conversational_generation_sft",
        "input_jsonl": str(ROOT / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_sft_cap50.jsonl"),
        "max_tokens": 512,
    },
    {
        "workload": "conversational_generation_gen",
        "input_jsonl": str(ROOT / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/conversational_generation_gen_cap50.jsonl"),
        "max_tokens": 512,
    },
    {
        "workload": "long_chain_reasoning",
        "input_jsonl": str(ROOT / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_chain_reasoning_cap50.jsonl"),
        "max_tokens": 1024,
    },
    {
        "workload": "long_context_completion",
        "input_jsonl": str(ROOT / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_context_completion_cap50.jsonl"),
        "max_tokens": 2048,
    },
    {
        "workload": "long_horizon_swe",
        "input_jsonl": str(ROOT / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/long_horizon_swe_cap50.jsonl"),
        "max_tokens": 1024,
    },
    {
        "workload": "mathematical_reasoning",
        "input_jsonl": str(ROOT / "data/benchmarks/By_split_phase_1/apex_paper_eval_cap50/mathematical_reasoning_cap50.jsonl"),
        "max_tokens": 512,
    },
]


def visible_gpu_for_worker(worker_idx: int) -> str:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [x.strip() for x in raw.split(",") if x.strip()]
    if visible and worker_idx < len(visible):
        return visible[worker_idx]
    return str(worker_idx)


def target_dir(workload: str, k: int) -> Path:
    return (
        FIXED_RUNS_ROOT
        / workload
        / "eagle3"
        / f"k{k}_temp0.0_eagle_Qwen3_8B_speculator.eagle3"
    )


def find_leaf_output_dir(tmp_dir: Path) -> Path:
    summaries = sorted(tmp_dir.rglob("summary.csv"))
    if not summaries:
        raise RuntimeError(f"No summary.csv found under {tmp_dir}")
    summaries = sorted(summaries, key=lambda p: len(p.parts), reverse=True)
    return summaries[0].parent


def copy_leaf_to_target(leaf: Path, target: Path, metadata: dict):
    backup = None

    if target.exists():
        backup = target.with_name(target.name + f"_BACKUP_before_eagle3_recommended_rerun_{int(time.time())}")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(target), str(backup))

    target.mkdir(parents=True, exist_ok=True)

    for item in leaf.iterdir():
        dst = target / item.name
        if item.is_file():
            shutil.copy2(item, dst)
        elif item.is_dir():
            shutil.copytree(item, dst)

    metadata["backup_dir"] = str(backup) if backup else ""
    (target / "_eagle3_recommended_rerun_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def run_one_k(workload_cfg: dict, k: int, gpu_id: str):
    workload = workload_cfg["workload"]
    tmp_dir = TMP_ROOT / workload / "eagle3_recommended" / f"k{k}"
    out_target = target_dir(workload, k)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_ROOT / f"{workload}_eagle3_recommended_k{k}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")

    cmd = [
        "python3",
        "src/layer0_data_collection/run_benchmark.py",
        "--workload", workload,
        "--run_name", f"{workload}_eagle3_recommended_k{k}_rerun",
        "--experiment", "baseline_matrix_all_methods_all_k",
        "--input_jsonl", workload_cfg["input_jsonl"],
        "--out_dir", str(tmp_dir),
        "--method", "eagle3",
        "--k", str(k),
        "--model", "Qwen/Qwen3-8B",
        "--eagle3_model", "RedHatAI/Qwen3-8B-speculator.eagle3",
        "--temperature", "0.0",
        "--top_p", "1.0",
        "--max_tokens", str(workload_cfg["max_tokens"]),
        "--limit", "0",
        "--gpu_memory_utilization", os.environ.get("APEXP_RERUN_GPU_MEM", "0.60"),
        "--enforce_eager",
        "--seed", "42",
        "--prompt_mode", "qwen3_chat_no_think",
    ]

    start = time.time()

    print(f"[START] workload={workload} k={k} gpu={gpu_id} prompt_mode=qwen3_chat_no_think", flush=True)
    print("[CMD]", " ".join(cmd), flush=True)

    with log_file.open("w") as lf:
        lf.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
        lf.write("prompt_mode=qwen3_chat_no_think\n")
        lf.write("[CMD] " + " ".join(cmd) + "\n\n")
        lf.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )

    end = time.time()

    if proc.returncode != 0:
        raise RuntimeError(
            f"FAILED workload={workload} k={k} gpu={gpu_id} "
            f"returncode={proc.returncode}. See {log_file}"
        )

    leaf = find_leaf_output_dir(tmp_dir)

    metadata = {
        "event": "eagle3_recommended_fixed_baseline_rerun",
        "workload": workload,
        "k": k,
        "input_jsonl": workload_cfg["input_jsonl"],
        "max_tokens": workload_cfg["max_tokens"],
        "gpu_id": str(gpu_id),
        "method": "eagle3",
        "target_model": "Qwen/Qwen3-8B",
        "eagle3_model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "prompt_mode": "qwen3_chat_no_think",
        "temperature": 0.0,
        "top_p": 1.0,
        "cmd": cmd,
        "tmp_dir": str(tmp_dir),
        "leaf_output_dir": str(leaf),
        "target_dir": str(out_target),
        "time_start": start,
        "time_end": end,
        "elapsed_s": end - start,
        "log_file": str(log_file),
        "paper_note": "EAGLE3 evaluated using Qwen3 chat template with thinking disabled.",
        "note": "This exact eagle3 fixed-run directory was overwritten by the recommended prompt-mode rerun.",
    }

    copy_leaf_to_target(leaf, out_target, metadata)

    print(f"[DONE] workload={workload} k={k} gpu={gpu_id}", flush=True)
    print(f"[TARGET] {out_target}", flush=True)


def worker_main(worker_idx: int, workload_cfg: dict):
    gpu_id = visible_gpu_for_worker(worker_idx)
    workload = workload_cfg["workload"]

    print("=" * 80, flush=True)
    print(f"[WORKER START] worker_idx={worker_idx} gpu_id={gpu_id} workload={workload}", flush=True)
    print("=" * 80, flush=True)

    for k in KS:
        run_one_k(workload_cfg, k, gpu_id)

    print("=" * 80, flush=True)
    print(f"[WORKER DONE] worker_idx={worker_idx} gpu_id={gpu_id} workload={workload}", flush=True)
    print("=" * 80, flush=True)


def main():
    print("ROOT:", ROOT, flush=True)
    print("FIXED_RUNS_ROOT:", FIXED_RUNS_ROOT, flush=True)
    print("TMP_ROOT:", TMP_ROOT, flush=True)
    print("LOG_ROOT:", LOG_ROOT, flush=True)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"), flush=True)

    for w in WORKLOADS:
        p = Path(w["input_jsonl"])
        if not p.exists():
            raise FileNotFoundError(f"Missing input JSONL for {w['workload']}: {p}")

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, w in enumerate(WORKLOADS):
        for k in KS:
            manifest.append({
                "worker_idx": i,
                "gpu_id": visible_gpu_for_worker(i),
                "workload": w["workload"],
                "k": k,
                "input_jsonl": w["input_jsonl"],
                "prompt_mode": "qwen3_chat_no_think",
                "target_dir": str(target_dir(w["workload"], k)),
            })

    manifest_path = FIXED_RUNS_ROOT.parent / "eagle3_recommended_rerun_7gpu_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("WROTE manifest:", manifest_path, flush=True)

    procs = []
    for i, w in enumerate(WORKLOADS):
        p = Process(target=worker_main, args=(i, w))
        p.start()
        procs.append((i, w["workload"], p))

    failed = []
    for i, workload, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((i, workload, p.exitcode))

    if failed:
        raise RuntimeError(f"Some EAGLE3 recommended workers failed: {failed}")

    print("[ALL DONE] EAGLE3 recommended rerun completed for all workloads and k values.", flush=True)


if __name__ == "__main__":
    main()
