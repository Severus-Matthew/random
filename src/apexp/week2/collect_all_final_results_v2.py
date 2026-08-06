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


def first_float(d: dict[str, Any], keys: list[str], default=None):
    for k in keys:
        if k in d:
            v = fnum(d.get(k), None)
            if v is not None:
                return v
    return default


def first_int(d: dict[str, Any], keys: list[str], default=None):
    for k in keys:
        if k in d:
            v = inum(d.get(k), None)
            if v is not None:
                return v
    return default


def norm_k(method, k):
    if method == "ar":
        return "NA"
    if k is None:
        return "NA"
    s = str(k)
    if s.lower() in {"", "nan", "none", "null"}:
        return "NA"
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_baselines(root: Path):
    rows = []
    trace_files = list(root.rglob("traces.jsonl"))
    print(f"[baseline] trace files found: {len(trace_files)}")

    for p in trace_files:
        for r in read_jsonl(p):
            workload = r.get("workload")
            method = r.get("method")
            if not workload or not method:
                continue

            method = str(method)
            k = norm_k(method, r.get("k"))

            runtime_s = first_float(r, ["latency_s", "wall_s", "elapsed_s", "runtime_s", "duration_s"])
            tps = first_float(r, ["tokens_per_sec", "target_tokens_per_sec", "actual_tps", "throughput_tps", "tps"])
            out_tokens = first_int(r, ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens"])

            if runtime_s is None or tps is None:
                continue

            if method == "ar":
                draft_tokens = 0.0
                accepted_tokens = 0.0
                rejected_tokens = 0.0
                system = "baseline_ar"
            else:
                draft_tokens = first_float(r, ["draft_tokens"], 0.0)
                accepted_tokens = first_float(r, ["accepted_tokens_total"], 0.0)
                rejected_tokens = first_float(r, ["rejected_tokens"], None)
                if rejected_tokens is None:
                    rejected_tokens = max(draft_tokens - accepted_tokens, 0.0)
                system = f"baseline_{method}_k{k}"

            rows.append({
                "run_family": "baseline",
                "system": system,
                "pool": "baseline",
                "workload": str(workload),
                "method": method,
                "k": k,
                "runtime_s": runtime_s,
                "tokens_per_sec": tps,
                "n_output_tokens": out_tokens,
                "draft_tokens": draft_tokens,
                "accepted_tokens_total": accepted_tokens,
                "rejected_tokens": rejected_tokens,
                "wasted_tokens": rejected_tokens,
                "prompt_hash": r.get("prompt_hash"),
                "source_index": r.get("request_ordinal", r.get("source_index")),
                "raw_file": str(p),
            })

    return rows


def aggregate_hot_blocks(root: Path):
    """
    Corrected hot trace aggregation.

    Critical fix:
      Use (pool, global_index) as the main key.
      Do NOT use task_id alone because task_id can repeat across pools.
    """
    by_pool_global = {}
    by_pool_hash = {}

    block_files = list(root.rglob("block_events.jsonl"))

    # Exclude stale gpu-level trace files if they exist.
    clean_block_files = []
    for p in block_files:
        if any(part.startswith("gpu") for part in p.parts):
            continue
        clean_block_files.append(p)

    print(f"[hot blocks] {root.name}: block files found={len(block_files)}, used={len(clean_block_files)}")

    event_count = 0
    missing_key = 0

    for p in clean_block_files:
        for e in read_jsonl(p):
            pool = e.get("pool")
            workload = e.get("workload")
            method = e.get("method")
            global_index = e.get("global_index")
            prompt_hash = e.get("prompt_hash")
            slow_k = e.get("slow_k")

            if not pool or global_index is None:
                missing_key += 1
                continue

            event_count += 1

            k_actual = first_float(e, ["k_actual", "scheduled_spec_len", "active_k"], 0.0)
            accepted_len = first_float(e, ["accepted_len", "num_accepted_tokens"], 0.0)

            rejected = first_float(e, ["num_rejected", "rejected_tokens"], None)
            if rejected is None:
                rejected = max(k_actual - accepted_len, 0.0)

            # Draft tokens per block should be accepted + rejected.
            # This avoids overcounting correction tokens.
            draft = max(accepted_len + rejected, 0.0)

            rec = {
                "draft_tokens": draft,
                "accepted_tokens_total": max(accepted_len, 0.0),
                "rejected_tokens": max(rejected, 0.0),
                "wasted_tokens": max(rejected, 0.0),
                "n_blocks": 1,
            }

            key = (str(pool), str(global_index))
            d = by_pool_global.setdefault(key, {
                "draft_tokens": 0.0,
                "accepted_tokens_total": 0.0,
                "rejected_tokens": 0.0,
                "wasted_tokens": 0.0,
                "n_blocks": 0,
            })
            for c in rec:
                d[c] += rec[c]

            if workload is not None and method is not None and prompt_hash is not None:
                hkey = (str(pool), str(workload), str(method), str(slow_k), str(prompt_hash))
                h = by_pool_hash.setdefault(hkey, {
                    "draft_tokens": 0.0,
                    "accepted_tokens_total": 0.0,
                    "rejected_tokens": 0.0,
                    "wasted_tokens": 0.0,
                    "n_blocks": 0,
                })
                for c in rec:
                    h[c] += rec[c]

    print(f"[hot blocks] {root.name}: events used={event_count}, missing main key={missing_key}")

    # Sanity print by pool directly from block events.
    sanity = {}
    for (pool, _gi), v in by_pool_global.items():
        d = sanity.setdefault(pool, {
            "draft_tokens": 0.0,
            "accepted_tokens_total": 0.0,
            "rejected_tokens": 0.0,
            "n_blocks": 0,
            "n_requests_with_blocks": 0,
        })
        d["draft_tokens"] += v["draft_tokens"]
        d["accepted_tokens_total"] += v["accepted_tokens_total"]
        d["rejected_tokens"] += v["rejected_tokens"]
        d["n_blocks"] += v["n_blocks"]
        d["n_requests_with_blocks"] += 1

    print(f"[hot blocks] {root.name}: sanity by pool")
    for pool, v in sorted(sanity.items()):
        rate = v["rejected_tokens"] / v["draft_tokens"] if v["draft_tokens"] else None
        print(
            f"  {pool:10s} requests_with_blocks={v['n_requests_with_blocks']} "
            f"blocks={v['n_blocks']} draft={v['draft_tokens']:.0f} "
            f"accepted={v['accepted_tokens_total']:.0f} rejected={v['rejected_tokens']:.0f} "
            f"rejection_rate={rate}"
        )

    return by_pool_global, by_pool_hash


def lookup_hot_agg(r, by_pool_global, by_pool_hash):
    pool = r.get("pool")
    global_index = r.get("global_index")

    if pool is not None and global_index is not None:
        key = (str(pool), str(global_index))
        if key in by_pool_global:
            return by_pool_global[key]

    workload = r.get("workload")
    method = r.get("method")
    slow_k = r.get("slow_k")
    prompt_hash = r.get("prompt_hash")

    if None not in [pool, workload, method, slow_k, prompt_hash]:
        key = (str(pool), str(workload), str(method), str(slow_k), str(prompt_hash))
        if key in by_pool_hash:
            return by_pool_hash[key]

    return None


def load_hot(root: Path, family: str):
    rows = []
    result_files = list((root / "results").glob("*.json"))
    print(f"[hot results] {family}: result files found={len(result_files)}")

    by_pool_global, by_pool_hash = aggregate_hot_blocks(root)

    missing_agg = 0

    for p in result_files:
        r = read_json(p)
        if not r:
            continue

        status = r.get("status")
        if status not in {None, "SUCCESS", "success"}:
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

        agg = lookup_hot_agg(r, by_pool_global, by_pool_hash)

        if agg is None:
            missing_agg += 1
            draft_tokens = 0.0
            accepted_tokens = 0.0
            rejected_tokens = 0.0
            n_blocks = 0
        else:
            draft_tokens = agg["draft_tokens"]
            accepted_tokens = agg["accepted_tokens_total"]
            rejected_tokens = agg["rejected_tokens"]
            n_blocks = agg["n_blocks"]

        k = norm_k(method, r.get("slow_k", r.get("k")))
        system = f"{family}_{pool}"

        rows.append({
            "run_family": family,
            "system": system,
            "pool": str(pool),
            "workload": str(workload),
            "method": str(method),
            "k": k,
            "runtime_s": runtime_s,
            "tokens_per_sec": tps,
            "n_output_tokens": out_tokens,
            "draft_tokens": draft_tokens,
            "accepted_tokens_total": accepted_tokens,
            "rejected_tokens": rejected_tokens,
            "wasted_tokens": rejected_tokens,
            "n_blocks": n_blocks,
            "prompt_hash": r.get("prompt_hash"),
            "source_index": r.get("source_index"),
            "global_index": r.get("global_index"),
            "raw_file": str(p),
        })

    print(f"[hot results] {family}: rows={len(rows)}, missing block aggregate={missing_agg}")
    return rows


def summarize(df: pd.DataFrame, group_cols: list[str]):
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
            mean_output_tokens=("n_output_tokens", "mean"),
            draft_tokens_total=("draft_tokens", "sum"),
            accepted_tokens_total=("accepted_tokens_total", "sum"),
            rejected_tokens_total=("rejected_tokens", "sum"),
            wasted_tokens_total=("wasted_tokens", "sum"),
            wasted_tokens_per_request=("wasted_tokens", "mean"),
            n_blocks_total=("n_blocks", "sum"),
        )
        .reset_index()
    )

    s["global_tps_output_tokens_over_runtime"] = s["output_tokens_total"] / s["total_runtime_s"]

    denom = s["draft_tokens_total"].where(s["draft_tokens_total"] != 0)
    s["overall_rejection_rate"] = s["rejected_tokens_total"] / denom
    s["overall_acceptance_rate"] = s["accepted_tokens_total"] / denom

    return s


def add_speedups_by_workload(s: pd.DataFrame):
    ar = s[s["system"] == "baseline_ar"][[
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

    out = s.merge(ar, on="workload", how="left")
    out["latency_speedup_vs_ar"] = out["ar_avg_runtime_s"] / out["avg_runtime_s"]
    out["mean_tps_speedup_vs_ar"] = out["mean_tps"] / out["ar_mean_tps"]
    out["global_tps_speedup_vs_ar"] = out["global_tps_output_tokens_over_runtime"] / out["ar_global_tps"]
    out["total_runtime_ratio_vs_ar"] = out["total_runtime_s"] / out["ar_total_runtime_s"]
    return out


def add_speedups_overall(s: pd.DataFrame):
    ar = s[s["system"] == "baseline_ar"]
    if len(ar) == 0:
        return s
    ar = ar.iloc[0]

    out = s.copy()
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
    rows += load_baselines(baseline_root)
    rows += load_hot(hot_default_root, "apex_default")
    rows += load_hot(hot_window2_root, "apex_window2")

    df = pd.DataFrame(rows)
    print(f"[all] raw rows: {len(df)}")

    if len(df) == 0:
        raise SystemExit("No rows collected.")

    # Conservative dedupe: do not use task_id; use real request identifiers.
    dedupe_cols = ["run_family", "system", "workload", "method", "k", "prompt_hash", "source_index", "global_index"]
    dedupe_cols = [c for c in dedupe_cols if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_cols, keep="first")
    print(f"[all] after dedupe: {len(df)} removed={before-len(df)}")

    df.to_csv(out_dir / "request_level_all_runs_v2.csv", index=False)

    by_workload = summarize(df, ["run_family", "system", "pool", "workload"])
    by_workload = add_speedups_by_workload(by_workload)
    by_workload.to_csv(out_dir / "summary_by_workload_v2.csv", index=False)

    overall = summarize(df, ["run_family", "system", "pool"])
    overall = add_speedups_overall(overall)
    overall.to_csv(out_dir / "summary_overall_v2.csv", index=False)

    by_method = summarize(df, ["run_family", "system", "pool", "workload", "method", "k"])
    by_method.to_csv(out_dir / "summary_by_workload_method_k_v2.csv", index=False)

    overall[[
        "run_family", "system", "pool", "n_requests",
        "total_runtime_s", "avg_runtime_s", "mean_tps",
        "global_tps_output_tokens_over_runtime",
        "latency_speedup_vs_ar", "mean_tps_speedup_vs_ar",
        "draft_tokens_total", "accepted_tokens_total",
        "rejected_tokens_total", "wasted_tokens_total",
        "wasted_tokens_per_request", "overall_rejection_rate",
        "overall_acceptance_rate",
    ]].to_csv(out_dir / "final_overall_table_v2.csv", index=False)

    by_workload[[
        "run_family", "system", "pool", "workload", "n_requests",
        "total_runtime_s", "avg_runtime_s", "mean_tps",
        "latency_speedup_vs_ar", "mean_tps_speedup_vs_ar",
        "draft_tokens_total", "accepted_tokens_total",
        "rejected_tokens_total", "wasted_tokens_total",
        "wasted_tokens_per_request", "overall_rejection_rate",
        "overall_acceptance_rate",
    ]].to_csv(out_dir / "final_by_workload_table_v2.csv", index=False)

    print("\nWROTE:")
    for name in [
        "request_level_all_runs_v2.csv",
        "summary_overall_v2.csv",
        "summary_by_workload_v2.csv",
        "summary_by_workload_method_k_v2.csv",
        "final_overall_table_v2.csv",
        "final_by_workload_table_v2.csv",
    ]:
        print(" ", out_dir / name)

    print("\n=== OVERALL v2 ===")
    cols = [
        "system", "n_requests", "total_runtime_s", "avg_runtime_s",
        "mean_tps", "mean_tps_speedup_vs_ar",
        "draft_tokens_total", "accepted_tokens_total",
        "rejected_tokens_total", "overall_rejection_rate",
    ]
    print(overall[cols].sort_values("mean_tps", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
