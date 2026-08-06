#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def slug(s: str) -> str:
    s = str(s)
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def model_slug(s: str) -> str:
    if not s:
        return ""
    return slug(s.split("/")[-1])


def get_default_draft_model(cfg: Dict[str, Any]) -> str:
    # Your config policy says Qwen/Qwen3-0.6B is the default meaningful draft.
    pol = cfg.get("_draft_model_policy", {})
    default_text = pol.get("default", "")
    if "Qwen/Qwen3-0.6B" in default_text:
        return "Qwen/Qwen3-0.6B"
    return "Qwen/Qwen3-0.6B"


def expand_experiment(cfg: Dict[str, Any], exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    g = cfg["global"]
    data_dir = Path(g["data_dir"])
    workload_files = cfg["workload_files"]
    workload_max_tokens = cfg.get("workload_max_tokens", {})
    default_draft = get_default_draft_model(cfg)

    if exp["workloads"] == "all":
        workloads = list(workload_files.keys())
    else:
        workloads = list(exp["workloads"])

    configs = exp["configs"]
    context_lengths = exp.get("context_lengths", [0])
    turn_depths = exp.get("turn_depths", [1])

    jobs = []
    for workload in workloads:
        if workload not in workload_files:
            raise KeyError(f"Unknown workload {workload} in experiment {exp['name']}")

        input_jsonl = data_dir / workload_files[workload]
        max_tokens = int(workload_max_tokens.get(workload, 512))

        for c in configs:
            method = c["method"]
            k = int(c.get("k", 1))
            temp = float(c.get("temperature", 0.0))

            # For ordinary experiments, context_lengths=[0], turn_depths=[1].
            # For F_context_growth, context_lengths expands --max_prompt_tokens.
            # For G_multi_turn, turn_depths expands --num_turns.
            for ctx in context_lengths:
                for turns in turn_depths:
                    draft_model = c.get("draft_model", "")
                    if method == "draft_sd" and not draft_model:
                        draft_model = default_draft

                    eagle3_model = c.get("eagle3_model", g.get("eagle3_model", ""))
                    target_model = c.get("target_model", g.get("target_model", "Qwen/Qwen3-8B"))

                    ngram_min = c.get("ngram_lookup_min", "")
                    ngram_max = c.get("ngram_lookup_max", "")

                    job = {
                        "source_experiment": exp["name"],
                        "phase": str(exp.get("phase", "")),
                        "workload": workload,
                        "input_jsonl": str(input_jsonl),
                        "method": method,
                        "k": k,
                        "temperature": temp,
                        "ngram_lookup_min": ngram_min,
                        "ngram_lookup_max": ngram_max,
                        "draft_model": draft_model,
                        "eagle3_model": eagle3_model,
                        "target_model": target_model,
                        "max_prompt_tokens": int(ctx),
                        "num_turns": int(turns),
                        "limit": int(g.get("limit", 100)),
                        "max_tokens": max_tokens,
                        "seed": int(g.get("seed", 42)),
                    }

                    jobs.append(job)

    return jobs


def dedupe_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate exact run-equivalent jobs while preserving source experiment labels.
    This avoids re-running the same workload/method/k/temp/context/turn/model config
    across A/B/H/I/K.
    """
    by_key: Dict[Tuple, Dict[str, Any]] = {}

    for j in jobs:
        key = (
            j["workload"],
            j["input_jsonl"],
            j["method"],
            j["k"],
            j["temperature"],
            j["ngram_lookup_min"],
            j["ngram_lookup_max"],
            j["draft_model"],
            j["eagle3_model"],
            j["target_model"],
            j["max_prompt_tokens"],
            j["num_turns"],
            j["limit"],
            j["max_tokens"],
            j["seed"],
        )

        if key not in by_key:
            jj = dict(j)
            jj["source_experiments"] = j["source_experiment"]
            by_key[key] = jj
        else:
            prev = by_key[key]
            exps = set(prev["source_experiments"].split(","))
            exps.add(j["source_experiment"])
            prev["source_experiments"] = ",".join(sorted(exps))

    return list(by_key.values())


def job_id(j: Dict[str, Any]) -> str:
    parts = [
        j["workload"],
        j["method"],
        f"k{j['k']}",
        f"t{j['temperature']}",
    ]

    if j["method"] == "ngram_sd":
        parts.append(f"ng{j['ngram_lookup_min']}-{j['ngram_lookup_max']}")

    if j["method"] == "draft_sd":
        parts.append(f"draft_{model_slug(j['draft_model'])}")

    if j.get("max_prompt_tokens", 0) > 0:
        parts.append(f"ctx{j['max_prompt_tokens']}")

    if j.get("num_turns", 1) > 1:
        parts.append(f"turns{j['num_turns']}")

    parts.append(f"seed{j['seed']}")

    return slug("_".join(parts))


def priority(j: Dict[str, Any]) -> int:
    """
    Lower number runs earlier.
    Core controller data first:
      0: AR baselines
      1: k=4 main methods
      2: k sweep for all speculative methods
      3: rejection/oracle variants
      4: temperature/context/multiturn/model-window extras
    """
    method = j["method"]
    k = int(j["k"])
    temp = float(j["temperature"])
    ctx = int(j.get("max_prompt_tokens", 0))
    turns = int(j.get("num_turns", 1))
    src = j.get("source_experiments", "")

    if method == "ar" and temp == 0.0 and ctx == 0 and turns == 1:
        return 0
    if k == 4 and temp == 0.0 and ctx == 0 and turns == 1 and method in {"ngram_sd", "draft_sd", "eagle3"}:
        return 1
    if temp == 0.0 and ctx == 0 and turns == 1 and method in {"ngram_sd", "draft_sd", "eagle3"}:
        return 2
    if any(x in src for x in ["H_oracle_k_study", "I_entropy_bucket_study", "K_rejection_study", "J_intra_sequence_transition"]):
        return 3
    return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--only-core", action="store_true",
                    help="Only A/B/H/I/K-like controller runs; skip C/D/E/F/G/J extras.")
    ap.add_argument("--limit-override", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = cfg["experiments"]
    if args.only_core:
        keep = {
            "A_baseline_sweep",
            "B_k_sweep",
            "H_oracle_k_study",
            "I_entropy_bucket_study",
            "K_rejection_study",
            "J_intra_sequence_transition",
        }
        experiments = [e for e in experiments if e["name"] in keep]

    all_jobs = []
    for exp in experiments:
        all_jobs.extend(expand_experiment(cfg, exp))

    if not args.no_dedupe:
        all_jobs = dedupe_jobs(all_jobs)

    if args.limit_override > 0:
        for j in all_jobs:
            j["limit"] = args.limit_override

    rows = []
    for j in all_jobs:
        jid = job_id(j)
        run_dir = Path(args.out_root) / "runs" / jid
        j["job_id"] = jid
        j["run_dir"] = str(run_dir)
        j["priority"] = priority(j)
        rows.append(j)

    rows.sort(key=lambda x: (x["priority"], x["workload"], x["method"], x["k"], x["temperature"], x["job_id"]))

    cols = [
        "priority",
        "job_id",
        "source_experiments",
        "phase",
        "workload",
        "input_jsonl",
        "method",
        "k",
        "temperature",
        "ngram_lookup_min",
        "ngram_lookup_max",
        "draft_model",
        "eagle3_model",
        "target_model",
        "max_prompt_tokens",
        "num_turns",
        "limit",
        "max_tokens",
        "seed",
        "run_dir",
    ]

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    def safe_cell(x):
        if x is None:
            return "NA"
        x = str(x)
        if x == "":
            return "NA"
        return x

    with manifest.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(safe_cell(r.get(c, "NA")) for c in cols) + "\n")

    # Write workload map
    with (out_root / "workload_map.tsv").open("w") as f:
        f.write("workload\tinput_jsonl\tmax_tokens\n")
        for w, fname in cfg["workload_files"].items():
            f.write(f"{w}\t{Path(cfg['global']['data_dir']) / fname}\t{cfg['workload_max_tokens'].get(w, '')}\n")

    # Job count summary
    counts = {}
    for r in rows:
        key = (r["workload"], r["method"])
        counts[key] = counts.get(key, 0) + 1

    print("CONFIG:", args.config)
    print("OUT_ROOT:", out_root)
    print("MANIFEST:", manifest)
    print("DEDUPED:", not args.no_dedupe)
    print("ONLY_CORE:", args.only_core)
    print("NUM_JOBS:", len(rows))
    print()
    print("WORKLOADS:")
    for w, fname in cfg["workload_files"].items():
        p = Path(cfg["global"]["data_dir"]) / fname
        print(f"  {w}: {p}")
    print()
    print("JOB COUNTS BY WORKLOAD/METHOD:")
    for (w, m), n in sorted(counts.items()):
        print(f"  {w:35s} {m:10s} {n}")
    print()
    print("Wrote:", manifest)
    print("Wrote:", out_root / "workload_map.tsv")


if __name__ == "__main__":
    main()
