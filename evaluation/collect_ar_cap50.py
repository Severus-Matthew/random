#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import pandas as pd


def iter_jsonl(path: Path):
    with path.open("r", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue

            # Robust to literal "\n" packed JSONL.
            parts = raw.split("\\n") if "}\\n{" in raw else [raw]

            for part in parts:
                part = part.strip()
                if not part:
                    continue
                try:
                    yield json.loads(part)
                except Exception:
                    continue


def as_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan")


def get_any(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def infer_workload_from_path(p: Path):
    parts = list(p.parts)
    if "fixed_runs" in parts:
        i = parts.index("fixed_runs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def trace_rows(trace_path: Path):
    workload = infer_workload_from_path(trace_path)
    run_dir = trace_path.parent
    rows = []

    for idx, obj in enumerate(iter_jsonl(trace_path)):
        # Common request-level metric names from our benchmark traces.
        tps = as_float(get_any(obj, [
            "tokens_per_sec",
            "tps",
            "throughput",
            "output_tokens_per_sec",
            "mean_tps",
        ]))

        latency = as_float(get_any(obj, [
            "latency_s",
            "wall_s",
            "elapsed_s",
            "total_time_s",
            "e2e_latency_engine_s",
            "generation_time_s",
        ]))

        out_tok = as_float(get_any(obj, [
            "n_output_tokens",
            "output_tokens",
            "num_output_tokens",
            "completion_tokens",
            "generated_tokens",
            "num_generated_tokens",
        ]))

        if not math.isfinite(tps) and math.isfinite(out_tok) and math.isfinite(latency) and latency > 0:
            tps = out_tok / latency

        prompt_id = get_any(obj, [
            "prompt_id",
            "id",
            "request_id",
            "sample_id",
        ], f"{workload}_{idx:06d}")

        row = {
            "workload": workload,
            "method": "ar",
            "k": 1,
            "baseline_config": "ar_k1",
            "temperature": 0.0,
            "prompt_id": prompt_id,
            "tokens_per_sec": tps,
            "latency_s": latency,
            "n_output_tokens": out_tok,
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "run_dir": str(run_dir),
            "trace_path": str(trace_path),
        }

        # Preserve extra request diagnostics when available.
        for c in [
            "mean_entropy",
            "entropy_bucket",
            "repetition_density",
            "ttft_s",
            "itl_s",
            "tpot_s",
            "prefix_cache_hit_rate",
            "kv_cache_usage_perc",
        ]:
            if c in obj:
                row[c] = obj[c]

        rows.append(row)

    return rows


def summary_rows(summary_path: Path):
    workload = infer_workload_from_path(summary_path)
    run_dir = summary_path.parent

    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return []

    if df.empty:
        return []

    # Keep only useful columns but do not assume exact schema.
    df["workload"] = df["workload"] if "workload" in df.columns else workload
    df["method"] = "ar"
    df["k"] = 1
    df["baseline_config"] = "ar_k1"
    df["temperature"] = 0.0
    df["run_dir"] = str(run_dir)
    df["summary_path"] = str(summary_path)

    rename = {}
    if "mean_tps" in df.columns and "tokens_per_sec" not in df.columns:
        rename["mean_tps"] = "tokens_per_sec"
    if "tps" in df.columns and "tokens_per_sec" not in df.columns:
        rename["tps"] = "tokens_per_sec"
    if "mean_latency_s" in df.columns and "latency_s" not in df.columns:
        rename["mean_latency_s"] = "latency_s"
    if "output_tokens" in df.columns and "n_output_tokens" not in df.columns:
        rename["output_tokens"] = "n_output_tokens"

    df = df.rename(columns=rename)

    for c, val in [
        ("tokens_per_sec", float("nan")),
        ("latency_s", float("nan")),
        ("n_output_tokens", float("nan")),
        ("draft_tokens", 0),
        ("accepted_tokens", 0),
        ("rejected_tokens", 0),
    ]:
        if c not in df.columns:
            df[c] = val

    return df.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-root",
        default="results/apexp_block/config_large_500/evaluation/paper_cap50_profile_kgrid",
    )
    args = ap.parse_args()

    root = Path(args.eval_root)
    fixed_root = root / "fixed_runs"
    out_dir = root / "fixed_baseline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Important: AR output is nested:
    # fixed_runs/<workload>/ar/ar_temp0.0/paper_cap50_ar/<workload>_full/ar/k1_temp0.0/traces.jsonl
    trace_paths = sorted(fixed_root.glob("*/ar/ar_temp0.0/**/traces.jsonl"))
    summary_paths = sorted(fixed_root.glob("*/ar/ar_temp0.0/**/summary.csv"))

    print(f"[found] AR traces:    {len(trace_paths)}")
    print(f"[found] AR summaries: {len(summary_paths)}")

    rows = []

    for tp in trace_paths:
        r = trace_rows(tp)
        print(f"[trace] {tp} -> {len(r)} rows")
        rows.extend(r)

    # Fallback: use summary rows only for workloads where trace parsing failed.
    have_workloads = {r["workload"] for r in rows}
    for sp in summary_paths:
        wl = infer_workload_from_path(sp)
        if wl in have_workloads:
            continue
        r = summary_rows(sp)
        print(f"[summary fallback] {sp} -> {len(r)} rows")
        rows.extend(r)

    if not rows:
        raise SystemExit("No AR rows collected. Check whether traces.jsonl has readable JSON objects.")

    df = pd.DataFrame(rows)

    for c in [
        "tokens_per_sec",
        "latency_s",
        "n_output_tokens",
        "draft_tokens",
        "accepted_tokens",
        "rejected_tokens",
    ]:
        if c not in df.columns:
            df[c] = float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["workload"] = df["workload"].astype(str).str.replace("_cap50", "", regex=False)
    df["method"] = "ar"
    df["k"] = 1
    df["baseline_config"] = "ar_k1"

    out_path = out_dir / "ar_request_rows.csv"
    df.to_csv(out_path, index=False)

    overall = df.groupby(["method", "k", "baseline_config"], dropna=False).agg(
        n_requests=("tokens_per_sec", "count"),
        mean_tps=("tokens_per_sec", "mean"),
        median_tps=("tokens_per_sec", "median"),
        p05_tps=("tokens_per_sec", lambda s: s.quantile(0.05)),
        p95_tps=("tokens_per_sec", lambda s: s.quantile(0.95)),
        output_tokens=("n_output_tokens", "sum"),
        mean_latency_s=("latency_s", "mean"),
    ).reset_index()

    by_workload = df.groupby(["method", "k", "baseline_config", "workload"], dropna=False).agg(
        n_requests=("tokens_per_sec", "count"),
        mean_tps=("tokens_per_sec", "mean"),
        median_tps=("tokens_per_sec", "median"),
        p05_tps=("tokens_per_sec", lambda s: s.quantile(0.05)),
        p95_tps=("tokens_per_sec", lambda s: s.quantile(0.95)),
        output_tokens=("n_output_tokens", "sum"),
        mean_latency_s=("latency_s", "mean"),
    ).reset_index()

    overall.to_csv(out_dir / "ar_overall_micro.csv", index=False)
    by_workload.to_csv(out_dir / "ar_by_workload_micro.csv", index=False)

    print(f"\n[wrote] {out_path}")
    print(f"[wrote] {out_dir / 'ar_overall_micro.csv'}")
    print(f"[wrote] {out_dir / 'ar_by_workload_micro.csv'}")

    print("\nAR overall:")
    print(overall.to_string(index=False))

    print("\nAR by workload:")
    print(by_workload.to_string(index=False))


if __name__ == "__main__":
    main()
