#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def load_results(root: Path) -> pd.DataFrame:
    rows = []
    for p in (root / "results").glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        rows.append({
            "file": str(p),
            "pool": r.get("pool"),
            "mode": r.get("mode"),
            "method": r.get("method"),
            "workload": r.get("workload"),
            "task_id": r.get("task_id"),
            "prompt_hash": r.get("prompt_hash"),
            "slow_k": r.get("slow_k"),
            "status": r.get("status"),
            "wall_s": safe_float(r.get("wall_s")),
            "output_tokens": safe_float(r.get("output_tokens")),
            "tokens_per_sec": safe_float(r.get("tokens_per_sec")),
        })
    return pd.DataFrame(rows)


def iter_block_events(root: Path):
    for p in root.rglob("block_events.jsonl"):
        # skip weird hidden/system paths if any
        try:
            with p.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    e["_path"] = str(p)
                    yield e
        except Exception:
            continue


def block_event_pool_from_path(path: str) -> str:
    parts = Path(path).parts
    if "slow_only" in parts:
        return "slow_only"
    if "slow_fast" in parts:
        return "slow_fast"
    return "unknown"


def summarize_blocks(root: Path) -> pd.DataFrame:
    rows = []
    for e in iter_block_events(root):
        path = e.get("_path", "")
        pool = e.get("pool") or block_event_pool_from_path(path)

        k_actual = e.get("k_actual", e.get("num_draft_tokens", e.get("k")))
        accepted_len = e.get("accepted_len", 0)
        num_rejected = e.get("num_rejected", None)

        try:
            k_actual = int(k_actual or 0)
        except Exception:
            k_actual = 0

        try:
            accepted_len = int(accepted_len or 0)
        except Exception:
            accepted_len = 0

        if num_rejected is None:
            num_rejected = max(0, k_actual - accepted_len)
        try:
            num_rejected = int(num_rejected or 0)
        except Exception:
            num_rejected = max(0, k_actual - accepted_len)

        if k_actual <= 0:
            continue

        rows.append({
            "pool": pool,
            "method": e.get("method"),
            "workload": e.get("workload"),
            "req_id": str(e.get("req_id", "")),
            "block_index": e.get("block_index", e.get("block")),
            "k_actual": k_actual,
            "accepted_len": accepted_len,
            "num_rejected": num_rejected,
            "full_accept": int(bool(e.get("full_accept", accepted_len >= k_actual))),
            "active_k": e.get("active_k", e.get("k_actual", k_actual)),
            "path": path,
        })

    return pd.DataFrame(rows)


def summarize_learned_trace(root: Path):
    counts = Counter()
    active = Counter()
    examples = []

    for p in root.rglob("online_fast_trace.jsonl"):
        try:
            with p.open() as f:
                for line in f:
                    for part in line.split("\\\\n"):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            e = json.loads(part)
                        except Exception:
                            continue

                        typ = e.get("type")
                    if typ == "learned_policy_init":
                        counts[("learned_policy_init", e.get("loaded"), e.get("error", ""))] += 1

                    if typ == "choose_active_k_for_scheduler":
                        active[(e.get("mode"), e.get("active_k"))] += 1

                    if typ == "observe_block_after_verify_update_future_k":
                        counts[("observe", e.get("learned_policy_used"), e.get("policy_error", ""))] += 1
                        if e.get("learned_policy_used") and len(examples) < 8:
                            examples.append(e)
        except Exception:
            continue

    return counts, active, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("ROOT:", root)
    print("OUT:", out_dir)

    # ------------------------------------------------------------------
    # Request-level summary
    # ------------------------------------------------------------------
    res = load_results(root)
    print("\n=== result files ===")
    print("n_results:", len(res))
    if len(res):
        print(res["pool"].value_counts(dropna=False))
        print(res["status"].value_counts(dropna=False))

    res.to_csv(out_dir / "request_results.csv", index=False)

    ok = res[res["status"].eq("SUCCESS")].copy()

    req_summary = (
        ok.groupby(["pool", "method", "workload"], dropna=False)
        .agg(
            n=("task_id", "count"),
            mean_tps=("tokens_per_sec", "mean"),
            median_tps=("tokens_per_sec", "median"),
            mean_wall_s=("wall_s", "mean"),
            median_wall_s=("wall_s", "median"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )

    overall_req = (
        ok.groupby(["pool"], dropna=False)
        .agg(
            n=("task_id", "count"),
            mean_tps=("tokens_per_sec", "mean"),
            median_tps=("tokens_per_sec", "median"),
            mean_wall_s=("wall_s", "mean"),
            median_wall_s=("wall_s", "median"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )

    if set(overall_req["pool"]) >= {"slow_only", "slow_fast"}:
        so_tps = float(overall_req.loc[overall_req.pool == "slow_only", "mean_tps"].iloc[0])
        sf_tps = float(overall_req.loc[overall_req.pool == "slow_fast", "mean_tps"].iloc[0])
        so_wall = float(overall_req.loc[overall_req.pool == "slow_only", "mean_wall_s"].iloc[0])
        sf_wall = float(overall_req.loc[overall_req.pool == "slow_fast", "mean_wall_s"].iloc[0])

        overall_req["speedup_vs_slow_only_tps"] = np.nan
        overall_req.loc[overall_req.pool == "slow_fast", "speedup_vs_slow_only_tps"] = sf_tps / so_tps if so_tps > 0 else np.nan

        overall_req["latency_speedup_vs_slow_only"] = np.nan
        overall_req.loc[overall_req.pool == "slow_fast", "latency_speedup_vs_slow_only"] = so_wall / sf_wall if sf_wall > 0 else np.nan

    req_summary.to_csv(out_dir / "request_summary_by_workload.csv", index=False)
    overall_req.to_csv(out_dir / "request_summary_overall.csv", index=False)

    print("\n=== REQUEST OVERALL ===")
    print(overall_req.to_string(index=False))

    print("\n=== REQUEST BY WORKLOAD ===")
    print(req_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # Block/token summary
    # ------------------------------------------------------------------
    blk = summarize_blocks(root)
    blk.to_csv(out_dir / "block_events_flat.csv", index=False)

    print("\n=== block events ===")
    print("n_blocks:", len(blk))
    if len(blk):
        print(blk["pool"].value_counts(dropna=False))

        block_summary = (
            blk.groupby(["pool", "method", "workload"], dropna=False)
            .agg(
                n_blocks=("accepted_len", "count"),
                draft_tokens=("k_actual", "sum"),
                accepted_tokens=("accepted_len", "sum"),
                rejected_tokens=("num_rejected", "sum"),
                mean_k=("k_actual", "mean"),
                mean_accepted_len=("accepted_len", "mean"),
                mean_rejected=("num_rejected", "mean"),
                full_accept_rate=("full_accept", "mean"),
            )
            .reset_index()
        )

        block_summary["wasted_token_rate"] = (
            block_summary["rejected_tokens"] /
            block_summary["draft_tokens"].replace(0, np.nan)
        )
        block_summary["accepted_per_block"] = (
            block_summary["accepted_tokens"] /
            block_summary["n_blocks"].replace(0, np.nan)
        )

        overall_block = (
            blk.groupby(["pool"], dropna=False)
            .agg(
                n_blocks=("accepted_len", "count"),
                draft_tokens=("k_actual", "sum"),
                accepted_tokens=("accepted_len", "sum"),
                rejected_tokens=("num_rejected", "sum"),
                mean_k=("k_actual", "mean"),
                mean_accepted_len=("accepted_len", "mean"),
                mean_rejected=("num_rejected", "mean"),
                full_accept_rate=("full_accept", "mean"),
            )
            .reset_index()
        )

        overall_block["wasted_token_rate"] = (
            overall_block["rejected_tokens"] /
            overall_block["draft_tokens"].replace(0, np.nan)
        )
        overall_block["accepted_per_block"] = (
            overall_block["accepted_tokens"] /
            overall_block["n_blocks"].replace(0, np.nan)
        )

        # Merge request speedups into overall token metrics.
        if len(overall_req):
            m = overall_req[["pool", "mean_tps"]].copy()
            ar = m[m["pool"].eq("slow_only")]
            if len(ar):
                base_tps = float(ar["mean_tps"].iloc[0])
                overall_block = overall_block.merge(m, on="pool", how="left")
                overall_block["tps_ratio_vs_slow_only"] = overall_block["mean_tps"] / base_tps
                overall_block["STE_vs_slow_only"] = overall_block["tps_ratio_vs_slow_only"] * (1.0 - overall_block["wasted_token_rate"])
                overall_block["ATE_vs_slow_only"] = (
                    overall_block["tps_ratio_vs_slow_only"]
                    * overall_block["accepted_per_block"]
                    * (1.0 - overall_block["wasted_token_rate"])
                )

        block_summary.to_csv(out_dir / "block_summary_by_workload.csv", index=False)
        overall_block.to_csv(out_dir / "block_summary_overall.csv", index=False)

        print("\n=== BLOCK OVERALL ===")
        print(overall_block.to_string(index=False))

        print("\n=== BLOCK BY WORKLOAD ===")
        print(block_summary.to_string(index=False))

        # Active k distribution.
        active_dist = (
            blk.groupby(["pool", "method", "active_k"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["pool", "method", "active_k"])
        )
        active_dist.to_csv(out_dir / "active_k_distribution.csv", index=False)
        print("\n=== ACTIVE K DISTRIBUTION ===")
        print(active_dist.to_string(index=False))

    # ------------------------------------------------------------------
    # Learned policy usage
    # ------------------------------------------------------------------
    counts, active, examples = summarize_learned_trace(root)

    print("\n=== LEARNED POLICY TRACE COUNTS ===")
    for k, v in counts.most_common(50):
        print(k, v)

    print("\n=== CHOOSE ACTIVE K TRACE COUNTS ===")
    for k, v in active.most_common(50):
        print(k, v)

    with (out_dir / "learned_policy_trace_counts.json").open("w") as f:
        json.dump({str(k): v for k, v in counts.items()}, f, indent=2, sort_keys=True)

    with (out_dir / "active_k_trace_counts.json").open("w") as f:
        json.dump({str(k): v for k, v in active.items()}, f, indent=2, sort_keys=True)

    with (out_dir / "learned_policy_examples.json").open("w") as f:
        json.dump(examples, f, indent=2, sort_keys=True)

    print("\n=== LEARNED POLICY EXAMPLES ===")
    for e in examples[:5]:
        print(json.dumps({
            "req_id": e.get("req_id"),
            "block": e.get("block"),
            "k_actual": e.get("k_actual"),
            "accepted_len": e.get("accepted_len"),
            "old_next_k": e.get("old_next_k"),
            "next_k": e.get("next_k_for_future_unscheduled_block"),
            "policy_runtime_ms": e.get("policy_runtime_ms"),
            "candidate_scores": e.get("candidate_scores"),
        }, indent=2)[:3000])

    print("\nWROTE SUMMARY TO:", out_dir)


if __name__ == "__main__":
    main()
