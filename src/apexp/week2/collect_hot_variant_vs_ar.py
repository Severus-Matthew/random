from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def fnum(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.lower() in {"", "nan", "none", "null"}:
            return default
        return float(x)
    except Exception:
        return default


def inum(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.lower() in {"", "nan", "none", "null"}:
            return default
        return int(float(x))
    except Exception:
        return default


def first_float(d, keys, default=None):
    for k in keys:
        if k in d:
            v = fnum(d.get(k), None)
            if v is not None:
                return v
    return default


def first_int(d, keys, default=None):
    for k in keys:
        if k in d:
            v = inum(d.get(k), None)
            if v is not None:
                return v
    return default


def load_hot_results(hot_root: Path, label: str):
    rows = []
    result_files = list((hot_root / "results").glob("*.json"))

    for p in result_files:
        r = read_json(p)
        if not r:
            continue
        if r.get("status") not in {None, "SUCCESS", "success"}:
            continue

        pool = r.get("pool")
        workload = r.get("workload")
        method = r.get("method")
        if not pool or not workload or not method:
            continue

        runtime_s = first_float(r, ["wall_s", "latency_s", "elapsed_s", "runtime_s", "duration_s"])
        tps = first_float(r, ["tokens_per_sec", "target_tokens_per_sec", "actual_tps", "throughput_tps", "tps"])
        out_tokens = first_int(r, ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens"])

        if runtime_s is None or tps is None:
            continue

        rows.append({
            "run_family": label,
            "system": f"{label}_{pool}",
            "pool": pool,
            "workload": workload,
            "method": method,
            "slow_k": r.get("slow_k"),
            "runtime_s": runtime_s,
            "tokens_per_sec": tps,
            "n_output_tokens": out_tokens,
            "prompt_hash": r.get("prompt_hash"),
            "source_index": r.get("source_index"),
            "global_index": r.get("global_index"),
            "result_file": str(p),
        })

    return pd.DataFrame(rows)


def aggregate_hot_blocks(hot_root: Path):
    by_pool_global = {}
    active_k_rows = []

    block_files = list(hot_root.rglob("block_events.jsonl"))
    clean_block_files = [
        p for p in block_files
        if not any(part.startswith("gpu") for part in p.parts)
    ]

    print(f"block files found={len(block_files)}, used={len(clean_block_files)}")

    for p in clean_block_files:
        for e in read_jsonl(p):
            pool = e.get("pool")
            workload = e.get("workload")
            method = e.get("method")
            global_index = e.get("global_index")

            if not pool or global_index is None:
                continue

            k_actual = first_float(e, ["k_actual", "scheduled_spec_len", "active_k"], 0.0)
            active_k = first_float(e, ["active_k", "k_actual", "scheduled_spec_len"], k_actual)
            accepted_len = first_float(e, ["accepted_len", "num_accepted_tokens"], 0.0)
            rejected = first_float(e, ["num_rejected", "rejected_tokens"], None)

            if rejected is None:
                rejected = max(k_actual - accepted_len, 0.0)

            # For hot block traces, draft tokens should be accepted + rejected.
            draft = max(accepted_len + rejected, 0.0)

            key = (str(pool), str(global_index))
            d = by_pool_global.setdefault(key, {
                "draft_tokens": 0.0,
                "accepted_tokens_total": 0.0,
                "rejected_tokens": 0.0,
                "wasted_tokens": 0.0,
                "n_blocks": 0,
            })

            d["draft_tokens"] += draft
            d["accepted_tokens_total"] += max(accepted_len, 0.0)
            d["rejected_tokens"] += max(rejected, 0.0)
            d["wasted_tokens"] += max(rejected, 0.0)
            d["n_blocks"] += 1

            active_k_rows.append({
                "pool": pool,
                "workload": workload,
                "method": method,
                "slow_k": e.get("slow_k"),
                "active_k": active_k,
                "k_actual": k_actual,
                "accepted_len": accepted_len,
                "num_rejected": rejected,
            })

    return by_pool_global, pd.DataFrame(active_k_rows)


def attach_block_waste(hot_df: pd.DataFrame, block_agg: dict):
    rows = []

    for _, r in hot_df.iterrows():
        key = (str(r["pool"]), str(r["global_index"]))
        agg = block_agg.get(key, None)

        d = r.to_dict()
        if agg is None:
            d.update({
                "draft_tokens": 0.0,
                "accepted_tokens_total": 0.0,
                "rejected_tokens": 0.0,
                "wasted_tokens": 0.0,
                "n_blocks": 0,
            })
        else:
            d.update(agg)

        rows.append(d)

    return pd.DataFrame(rows)


def load_ar_reference(baseline_root: Path, hot_df: pd.DataFrame):
    needed = set()
    for _, r in hot_df.iterrows():
        if pd.notna(r.get("workload")) and pd.notna(r.get("source_index")):
            needed.add((str(r["workload"]), int(r["source_index"])))

    rows = []

    for p in baseline_root.rglob("traces.jsonl"):
        for r in read_jsonl(p):
            if r.get("method") != "ar":
                continue

            workload = r.get("workload")
            source_index = r.get("request_ordinal", r.get("source_index"))

            if workload is None or source_index is None:
                continue

            try:
                key = (str(workload), int(source_index))
            except Exception:
                continue

            # Match the same prompt subset as the hot run when possible.
            if needed and key not in needed:
                continue

            runtime_s = first_float(r, ["latency_s", "wall_s", "elapsed_s", "runtime_s", "duration_s"])
            tps = first_float(r, ["tokens_per_sec", "target_tokens_per_sec", "actual_tps", "throughput_tps", "tps"])
            out_tokens = first_int(r, ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens"])

            if runtime_s is None or tps is None:
                continue

            rows.append({
                "run_family": "baseline",
                "system": "baseline_ar_ref",
                "pool": "baseline",
                "workload": workload,
                "method": "ar",
                "slow_k": "NA",
                "runtime_s": runtime_s,
                "tokens_per_sec": tps,
                "n_output_tokens": out_tokens,
                "prompt_hash": r.get("prompt_hash"),
                "source_index": int(source_index),
                "global_index": None,
                "draft_tokens": 0.0,
                "accepted_tokens_total": 0.0,
                "rejected_tokens": 0.0,
                "wasted_tokens": 0.0,
                "n_blocks": 0,
                "result_file": str(p),
            })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols):
    s = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_requests=("runtime_s", "count"),
            total_runtime_s=("runtime_s", "sum"),
            avg_runtime_s=("runtime_s", "mean"),
            median_runtime_s=("runtime_s", "median"),
            mean_tps=("tokens_per_sec", "mean"),
            median_tps=("tokens_per_sec", "median"),
            output_tokens_total=("n_output_tokens", "sum"),
            draft_tokens_total=("draft_tokens", "sum"),
            accepted_tokens_total=("accepted_tokens_total", "sum"),
            rejected_tokens_total=("rejected_tokens", "sum"),
            wasted_tokens_total=("wasted_tokens", "sum"),
            n_blocks_total=("n_blocks", "sum"),
        )
        .reset_index()
    )

    s["wasted_token_pct"] = 100.0 * s["rejected_tokens_total"] / s["draft_tokens_total"].where(s["draft_tokens_total"] != 0)
    s["global_tps"] = s["output_tokens_total"] / s["total_runtime_s"]

    return s


def add_speedups(summary: pd.DataFrame):
    if "workload" in summary.columns:
        ar = summary[summary["system"] == "baseline_ar_ref"][[
            "workload", "mean_tps", "avg_runtime_s", "total_runtime_s", "global_tps"
        ]].rename(columns={
            "mean_tps": "ar_mean_tps",
            "avg_runtime_s": "ar_avg_runtime_s",
            "total_runtime_s": "ar_total_runtime_s",
            "global_tps": "ar_global_tps",
        })

        out = summary.merge(ar, on="workload", how="left")
    else:
        ar = summary[summary["system"] == "baseline_ar_ref"]
        out = summary.copy()
        if len(ar):
            ar = ar.iloc[0]
            out["ar_mean_tps"] = ar["mean_tps"]
            out["ar_avg_runtime_s"] = ar["avg_runtime_s"]
            out["ar_total_runtime_s"] = ar["total_runtime_s"]
            out["ar_global_tps"] = ar["global_tps"]

    out["mean_tps_speedup_vs_ar"] = out["mean_tps"] / out["ar_mean_tps"]
    out["latency_speedup_vs_ar"] = out["ar_avg_runtime_s"] / out["avg_runtime_s"]
    out["global_tps_speedup_vs_ar"] = out["global_tps"] / out["ar_global_tps"]
    out["total_runtime_ratio_vs_ar"] = out["total_runtime_s"] / out["ar_total_runtime_s"]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot-root", required=True)
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--label", default="apex_window2_mink4")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    hot_root = Path(args.hot_root)
    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hot = load_hot_results(hot_root, args.label)
    print("hot result rows:", len(hot))

    block_agg, active_k_df = aggregate_hot_blocks(hot_root)
    hot = attach_block_waste(hot, block_agg)

    ar = load_ar_reference(baseline_root, hot)
    print("matched AR reference rows:", len(ar))

    all_df = pd.concat([hot, ar], ignore_index=True)
    all_df.to_csv(out_dir / "request_level_window2_mink4_vs_ar.csv", index=False)
    active_k_df.to_csv(out_dir / "active_k_distribution_raw.csv", index=False)

    overall = summarize(all_df, ["run_family", "system", "pool"])
    overall = add_speedups(overall)
    overall.to_csv(out_dir / "summary_overall_window2_mink4.csv", index=False)

    by_workload = summarize(all_df, ["run_family", "system", "pool", "workload"])
    by_workload = add_speedups(by_workload)
    by_workload.to_csv(out_dir / "summary_by_workload_window2_mink4.csv", index=False)

    active_summary = (
        active_k_df
        .groupby(["pool", "workload", "method", "slow_k", "active_k"], dropna=False)
        .size()
        .reset_index(name="n_blocks")
        .sort_values(["pool", "workload", "method", "slow_k", "active_k"])
    )
    active_summary.to_csv(out_dir / "active_k_distribution_summary.csv", index=False)

    final_cols = [
        "system", "n_requests", "total_runtime_s", "avg_runtime_s", "mean_tps",
        "mean_tps_speedup_vs_ar", "latency_speedup_vs_ar",
        "draft_tokens_total", "accepted_tokens_total", "rejected_tokens_total",
        "wasted_token_pct", "n_blocks_total",
    ]

    print("\n=== OVERALL ===")
    print(overall[final_cols].sort_values("mean_tps", ascending=False).to_string(index=False))

    print("\n=== BY WORKLOAD ===")
    print(by_workload[["workload"] + final_cols].sort_values(["workload", "mean_tps"], ascending=[True, False]).to_string(index=False))

    print("\n=== ACTIVE K SUMMARY ===")
    print(active_summary.to_string(index=False))

    for name in [
        "request_level_window2_mink4_vs_ar.csv",
        "summary_overall_window2_mink4.csv",
        "summary_by_workload_window2_mink4.csv",
        "active_k_distribution_raw.csv",
        "active_k_distribution_summary.csv",
    ]:
        print("wrote:", out_dir / name)


if __name__ == "__main__":
    main()
