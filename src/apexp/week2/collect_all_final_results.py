from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def as_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.lower() in {"nan", "none", ""}:
            return default
        return float(x)
    except Exception:
        return default


def as_int(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.lower() in {"nan", "none", ""}:
            return default
        return int(float(x))
    except Exception:
        return default


def first_float(row: dict[str, Any], keys: list[str], default=None):
    for k in keys:
        if k in row:
            v = as_float(row.get(k), None)
            if v is not None:
                return v
    return default


def first_int(row: dict[str, Any], keys: list[str], default=None):
    for k in keys:
        if k in row:
            v = as_int(row.get(k), None)
            if v is not None:
                return v
    return default


def norm_k(method, k):
    if method == "ar":
        return "NA"
    if k is None or str(k).lower() in {"nan", "none", ""}:
        return "NA"
    return str(int(float(k))) if str(k).replace(".", "", 1).isdigit() else str(k)


def load_baseline_rows(root: Path):
    out = []

    trace_files = list(root.rglob("traces.jsonl"))
    print(f"[baseline] traces.jsonl files found: {len(trace_files)}")

    for p in trace_files:
        for r in read_jsonl(p):
            workload = r.get("workload")
            method = r.get("method")
            if not workload or not method:
                continue

            method = str(method)
            k = norm_k(method, r.get("k"))

            latency_s = first_float(r, ["latency_s", "wall_s", "elapsed_s", "runtime_s", "duration_s"])
            tps = first_float(r, ["tokens_per_sec", "target_tokens_per_sec", "actual_tps", "throughput_tps", "tps"])
            out_tokens = first_int(r, ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens", "completion_tokens"])

            draft_tokens = first_float(r, ["draft_tokens"], 0.0)
            accepted_tokens = first_float(r, ["accepted_tokens_total"], 0.0)

            rejected_tokens = first_float(r, ["rejected_tokens"], None)
            if rejected_tokens is None and method != "ar":
                rejected_tokens = max(draft_tokens - accepted_tokens, 0.0)
            if method == "ar":
                rejected_tokens = 0.0
                draft_tokens = 0.0
                accepted_tokens = 0.0

            if latency_s is None or tps is None:
                continue

            system = "baseline_ar" if method == "ar" else f"baseline_{method}_k{k}"

            out.append({
                "run_family": "baseline",
                "system": system,
                "pool": "baseline",
                "workload": workload,
                "method": method,
                "k": k,
                "runtime_s": latency_s,
                "tokens_per_sec": tps,
                "n_output_tokens": out_tokens,
                "draft_tokens": draft_tokens,
                "accepted_tokens_total": accepted_tokens,
                "rejected_tokens": rejected_tokens,
                "wasted_tokens": rejected_tokens,
                "acceptance_rate": first_float(r, ["acceptance_rate"], None),
                "prompt_hash": r.get("prompt_hash"),
                "source_index": r.get("request_ordinal", r.get("source_index")),
                "trace_file": str(p),
                "raw_source": "baseline_traces",
            })

    return out


def aggregate_hot_block_traces(root: Path):
    """
    Aggregates per-request block_events.jsonl files so we can estimate wasted/rejected
    tokens for slow_only and slow_fast.
    """
    agg_by_task = {}
    agg_by_global = {}
    agg_by_hash = {}

    block_files = list(root.rglob("block_events.jsonl"))
    print(f"[hot traces] {root.name}: block_events.jsonl files found: {len(block_files)}")

    for p in block_files:
        for e in read_jsonl(p):
            workload = e.get("workload")
            pool = e.get("pool")
            method = e.get("method")
            if not workload or not pool or not method:
                continue

            k_actual = first_float(e, ["k_actual", "scheduled_spec_len", "active_k"], 0.0)
            accepted_len = first_float(e, ["accepted_len", "num_accepted_tokens"], 0.0)
            rejected = first_float(e, ["num_rejected", "rejected_tokens"], None)

            if rejected is None:
                rejected = max(k_actual - accepted_len, 0.0)

            rec = {
                "draft_tokens": max(k_actual, 0.0),
                "accepted_tokens_total": max(accepted_len, 0.0),
                "rejected_tokens": max(rejected, 0.0),
                "wasted_tokens": max(rejected, 0.0),
                "n_blocks": 1,
            }

            task_id = e.get("task_id") or e.get("control_id")
            global_index = e.get("global_index")
            prompt_hash = e.get("prompt_hash")
            slow_k = e.get("slow_k")

            keys = []

            if task_id is not None:
                keys.append(("task", str(task_id)))

            if global_index is not None:
                keys.append(("global", str(pool), str(global_index)))

            if prompt_hash is not None:
                keys.append(("hash", str(pool), str(workload), str(method), str(slow_k), str(prompt_hash)))

            for key in keys:
                if key[0] == "task":
                    d = agg_by_task.setdefault(key[1], {"draft_tokens": 0.0, "accepted_tokens_total": 0.0, "rejected_tokens": 0.0, "wasted_tokens": 0.0, "n_blocks": 0})
                elif key[0] == "global":
                    d = agg_by_global.setdefault(key[1:], {"draft_tokens": 0.0, "accepted_tokens_total": 0.0, "rejected_tokens": 0.0, "wasted_tokens": 0.0, "n_blocks": 0})
                else:
                    d = agg_by_hash.setdefault(key[1:], {"draft_tokens": 0.0, "accepted_tokens_total": 0.0, "rejected_tokens": 0.0, "wasted_tokens": 0.0, "n_blocks": 0})

                for c in ["draft_tokens", "accepted_tokens_total", "rejected_tokens", "wasted_tokens", "n_blocks"]:
                    d[c] += rec[c]

    return agg_by_task, agg_by_global, agg_by_hash


def lookup_hot_trace_agg(r, agg_by_task, agg_by_global, agg_by_hash):
    task_id = r.get("task_id") or r.get("control_id")
    if task_id is not None and str(task_id) in agg_by_task:
        return agg_by_task[str(task_id)]

    pool = r.get("pool")
    global_index = r.get("global_index")
    if pool is not None and global_index is not None:
        key = (str(pool), str(global_index))
        if key in agg_by_global:
            return agg_by_global[key]

    workload = r.get("workload")
    method = r.get("method")
    slow_k = r.get("slow_k")
    prompt_hash = r.get("prompt_hash")
    if None not in [pool, workload, method, slow_k, prompt_hash]:
        key = (str(pool), str(workload), str(method), str(slow_k), str(prompt_hash))
        if key in agg_by_hash:
            return agg_by_hash[key]

    return None


def load_hot_rows(root: Path, run_family: str):
    out = []
    result_files = list((root / "results").glob("*.json"))
    print(f"[hot results] {run_family}: result json files found: {len(result_files)}")

    agg_by_task, agg_by_global, agg_by_hash = aggregate_hot_block_traces(root)

    missing_trace_agg = 0

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
        out_tokens = first_int(r, ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens", "completion_tokens"])

        if runtime_s is None or tps is None:
            continue

        trace_agg = lookup_hot_trace_agg(r, agg_by_task, agg_by_global, agg_by_hash)

        if trace_agg is None:
            missing_trace_agg += 1
            draft_tokens = first_float(r, ["draft_tokens"], 0.0)
            accepted_tokens = first_float(r, ["accepted_tokens_total"], 0.0)
            rejected_tokens = first_float(r, ["rejected_tokens", "wasted_tokens"], 0.0)
            n_blocks = None
        else:
            draft_tokens = trace_agg["draft_tokens"]
            accepted_tokens = trace_agg["accepted_tokens_total"]
            rejected_tokens = trace_agg["rejected_tokens"]
            n_blocks = trace_agg["n_blocks"]

        k = norm_k(method, r.get("slow_k", r.get("k")))
        system = f"{run_family}_{pool}"

        out.append({
            "run_family": run_family,
            "system": system,
            "pool": pool,
            "workload": workload,
            "method": method,
            "k": k,
            "runtime_s": runtime_s,
            "tokens_per_sec": tps,
            "n_output_tokens": out_tokens,
            "draft_tokens": draft_tokens,
            "accepted_tokens_total": accepted_tokens,
            "rejected_tokens": rejected_tokens,
            "wasted_tokens": rejected_tokens,
            "acceptance_rate": None if draft_tokens <= 0 else accepted_tokens / draft_tokens,
            "n_blocks": n_blocks,
            "prompt_hash": r.get("prompt_hash"),
            "source_index": r.get("source_index"),
            "global_index": r.get("global_index"),
            "result_file": str(p),
            "raw_source": "hot_results_plus_block_traces",
        })

    print(f"[hot results] {run_family}: rows={len(out)}, missing trace aggregates={missing_trace_agg}")
    return out


def summarize(df: pd.DataFrame, group_cols: list[str]):
    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_requests=("runtime_s", "count"),
            total_runtime_s=("runtime_s", "sum"),
            avg_runtime_s=("runtime_s", "mean"),
            median_runtime_s=("runtime_s", "median"),
            mean_tps=("tokens_per_sec", "mean"),
            median_tps=("tokens_per_sec", "median"),
            output_tokens_total=("n_output_tokens", "sum"),
            mean_output_tokens=("n_output_tokens", "mean"),
            draft_tokens_total=("draft_tokens", "sum"),
            accepted_tokens_total=("accepted_tokens_total", "sum"),
            rejected_tokens_total=("rejected_tokens", "sum"),
            wasted_tokens_total=("wasted_tokens", "sum"),
            wasted_tokens_per_request=("wasted_tokens", "mean"),
        )
        .reset_index()
    )

    g["global_tps_output_tokens_over_runtime"] = g["output_tokens_total"] / g["total_runtime_s"]
    g["overall_rejection_rate"] = g["rejected_tokens_total"] / g["draft_tokens_total"].replace({0.0: pd.NA})
    g["accepted_per_draft_overall"] = g["accepted_tokens_total"] / g["draft_tokens_total"].replace({0.0: pd.NA})
    return g


def add_speedups_by_workload(summary_by_workload: pd.DataFrame):
    ar = summary_by_workload[summary_by_workload["system"] == "baseline_ar"].copy()
    ar = ar[[
        "workload",
        "avg_runtime_s",
        "total_runtime_s",
        "mean_tps",
        "global_tps_output_tokens_over_runtime",
    ]].rename(columns={
        "avg_runtime_s": "ar_avg_runtime_s",
        "total_runtime_s": "ar_total_runtime_s",
        "mean_tps": "ar_mean_tps",
        "global_tps_output_tokens_over_runtime": "ar_global_tps",
    })

    out = summary_by_workload.merge(ar, on="workload", how="left")

    out["latency_speedup_vs_ar"] = out["ar_avg_runtime_s"] / out["avg_runtime_s"]
    out["mean_tps_speedup_vs_ar"] = out["mean_tps"] / out["ar_mean_tps"]
    out["global_tps_speedup_vs_ar"] = out["global_tps_output_tokens_over_runtime"] / out["ar_global_tps"]
    out["total_runtime_ratio_vs_ar"] = out["total_runtime_s"] / out["ar_total_runtime_s"]

    return out


def add_speedups_overall(summary_overall: pd.DataFrame):
    ar = summary_overall[summary_overall["system"] == "baseline_ar"]
    if len(ar) == 0:
        return summary_overall

    ar = ar.iloc[0]
    out = summary_overall.copy()

    out["ar_avg_runtime_s"] = ar["avg_runtime_s"]
    out["ar_total_runtime_s"] = ar["total_runtime_s"]
    out["ar_mean_tps"] = ar["mean_tps"]
    out["ar_global_tps"] = ar["global_tps_output_tokens_over_runtime"]

    out["latency_speedup_vs_ar"] = out["ar_avg_runtime_s"] / out["avg_runtime_s"]
    out["mean_tps_speedup_vs_ar"] = out["mean_tps"] / out["ar_mean_tps"]
    out["global_tps_speedup_vs_ar"] = out["global_tps_output_tokens_over_runtime"] / out["ar_global_tps"]
    out["total_runtime_ratio_vs_ar"] = out["total_runtime_s"] / out["ar_total_runtime_s"]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--hot-default-root", required=True)
    ap.add_argument("--hot-window2-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    baseline_root = Path(args.baseline_root)
    hot_default_root = Path(args.hot_default_root)
    hot_window2_root = Path(args.hot_window2_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += load_baseline_rows(baseline_root)
    rows += load_hot_rows(hot_default_root, "apex_default")
    rows += load_hot_rows(hot_window2_root, "apex_window2")

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise SystemExit("No rows collected. Check input paths.")

    before = len(df)
    dedupe_cols = [
        "run_family", "system", "workload", "method", "k",
        "prompt_hash", "source_index", "global_index",
    ]
    existing = [c for c in dedupe_cols if c in df.columns]
    df = df.drop_duplicates(subset=existing, keep="first")
    after = len(df)

    print(f"[dedupe] before={before}, after={after}, removed={before-after}")

    df.to_csv(out_dir / "request_level_all_runs.csv", index=False)

    summary_by_workload = summarize(df, ["run_family", "system", "pool", "workload"])
    summary_by_workload = add_speedups_by_workload(summary_by_workload)
    summary_by_workload.to_csv(out_dir / "summary_by_workload.csv", index=False)

    summary_overall = summarize(df, ["run_family", "system", "pool"])
    summary_overall = add_speedups_overall(summary_overall)
    summary_overall.to_csv(out_dir / "summary_overall.csv", index=False)

    # More detailed baseline/router-action summary.
    summary_by_workload_method_k = summarize(df, ["run_family", "system", "pool", "workload", "method", "k"])
    summary_by_workload_method_k.to_csv(out_dir / "summary_by_workload_method_k.csv", index=False)

    # Total runtime and wasted tokens quick views.
    total_runtime = summary_overall[[
        "run_family", "system", "pool", "n_requests", "total_runtime_s",
        "avg_runtime_s", "mean_tps", "global_tps_output_tokens_over_runtime",
        "latency_speedup_vs_ar", "mean_tps_speedup_vs_ar",
        "global_tps_speedup_vs_ar", "total_runtime_ratio_vs_ar",
    ]].sort_values("total_runtime_s")
    total_runtime.to_csv(out_dir / "total_runtime_overall.csv", index=False)

    wasted_overall = summary_overall[[
        "run_family", "system", "pool", "n_requests",
        "draft_tokens_total", "accepted_tokens_total",
        "rejected_tokens_total", "wasted_tokens_total",
        "wasted_tokens_per_request", "overall_rejection_rate",
    ]].sort_values("wasted_tokens_total")
    wasted_overall.to_csv(out_dir / "wasted_tokens_overall.csv", index=False)

    wasted_by_workload = summary_by_workload[[
        "run_family", "system", "pool", "workload", "n_requests",
        "draft_tokens_total", "accepted_tokens_total",
        "rejected_tokens_total", "wasted_tokens_total",
        "wasted_tokens_per_request", "overall_rejection_rate",
    ]].sort_values(["workload", "wasted_tokens_total"])
    wasted_by_workload.to_csv(out_dir / "wasted_tokens_by_workload.csv", index=False)

    print("\nWROTE:")
    for p in [
        "request_level_all_runs.csv",
        "summary_by_workload.csv",
        "summary_overall.csv",
        "summary_by_workload_method_k.csv",
        "total_runtime_overall.csv",
        "wasted_tokens_overall.csv",
        "wasted_tokens_by_workload.csv",
    ]:
        print(" ", out_dir / p)

    print("\n=== summary overall ===")
    cols = [
        "system", "n_requests", "total_runtime_s", "avg_runtime_s",
        "mean_tps", "global_tps_output_tokens_over_runtime",
        "mean_tps_speedup_vs_ar", "latency_speedup_vs_ar",
        "wasted_tokens_total", "overall_rejection_rate",
    ]
    print(summary_overall[cols].sort_values("mean_tps", ascending=False).to_string(index=False))

    print("\n=== by workload ===")
    cols2 = [
        "system", "workload", "n_requests", "total_runtime_s",
        "avg_runtime_s", "mean_tps", "mean_tps_speedup_vs_ar",
        "wasted_tokens_total", "overall_rejection_rate",
    ]
    print(summary_by_workload[cols2].sort_values(["workload", "mean_tps"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
