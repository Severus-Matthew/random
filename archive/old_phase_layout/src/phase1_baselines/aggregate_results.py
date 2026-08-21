#!/usr/bin/env python
"""
aggregate_results.py

Walks the results directory recursively, collects all summary.csv files,
and produces:
  - all_runs.csv              : flat concat of every summary.csv
  - aggregate_by_workload_method.csv : grouped stats
  - aggregate_by_run.csv      : same but also split by run_name (50_samples_k_sweep etc.)

Expected structure (from run_benchmark.py):
  results/<run_name>/<workload_or_run_name>/<method>/k<N>/summary.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_all_summaries(results_dir: Path) -> pd.DataFrame:
    paths = sorted(results_dir.glob("**/summary.csv"))
    if not paths:
        raise SystemExit(f"No summary.csv files found under {results_dir}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)

        # Enrich with path-derived metadata so we can group even if
        # columns are missing from older runs
        parts = p.parts  # …/results/run_name/workload/method/kN/summary.csv
        if len(parts) >= 5:
            df["_path_run_name"] = parts[-5]
            df["_path_workload"] = parts[-4]
            df["_path_method"]   = parts[-3]
            df["_path_k"]        = parts[-2]   # e.g. "k4"

        df["_source_file"] = str(p)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Prefer columns from the CSV; fall back to path-derived values
    if "run_name" not in combined.columns:
        combined["run_name"] = combined.get("_path_run_name", "unknown")
    if "workload" not in combined.columns:
        combined["workload"] = combined.get("_path_workload", "unknown")
    if "method" not in combined.columns:
        combined["method"] = combined.get("_path_method", "unknown")
    if "k" not in combined.columns:
        combined["k"] = combined["_path_k"].str.replace("k", "", regex=False).astype(float)

    # Drop internal helper columns
    combined.drop(columns=[c for c in combined.columns if c.startswith("_path_")],
                  inplace=True, errors="ignore")
    return combined


AGG_SPEC = {
    # bookkeeping
    "prompts":                                ("id",                               "count"),
    # core inference
    "latency_mean":                           ("latency_s",                        "mean"),
    "latency_p50":                            ("latency_s",                        "median"),
    "latency_p95":                            ("latency_s",                        lambda x: x.quantile(0.95)),
    "latency_std":                            ("latency_s",                        "std"),
    "tps_mean":                               ("tokens_per_sec",                   "mean"),
    "tps_p50":                                ("tokens_per_sec",                   "median"),
    "tps_std":                                ("tokens_per_sec",                   "std"),
    "n_output_tokens_mean":                   ("n_output_tokens",                  "mean"),
    # acceptance
    "acceptance_rate_mean":                   ("acceptance_rate",                  "mean"),
    "acceptance_rate_std":                    ("acceptance_rate",                  "std"),
    "accepted_tokens_per_verifier_pass_mean": ("accepted_tokens_per_verifier_pass","mean"),
    # phase 1 required
    "rollback_frequency_mean":                ("rollback_frequency",               "mean"),
    "rollback_frequency_std":                 ("rollback_frequency",               "std"),
    "verifier_utilization_mean":              ("verifier_utilization",             "mean"),
    # structural / token-regime
    "entropy_mean":                           ("mean_entropy",                     "mean"),
    "entropy_std":                            ("mean_entropy",                     "std"),
    "repetition_density_mean":                ("repetition_density",               "mean"),
    "acceptance_volatility_mean":             ("acceptance_volatility",            "mean"),
    "rejection_locality_mean":                ("rejection_locality",               "mean"),
}


def agg_group(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    # only keep columns that actually exist
    existing = {k: v for k, v in AGG_SPEC.items()
                if v[0] in df.columns}
    rename = {v[0]: k for k, v in existing.items()}
    funcs  = {v[0]: v[1] for k, v in existing.items()}
    result = df.groupby(group_cols, dropna=False).agg(funcs).reset_index()
    result.columns = group_cols + [rename.get(c, c) for c in result.columns[len(group_cols):]]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir",     default=None,
                    help="Where to write outputs (default: same as results_dir)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {results_dir} ...")
    df = load_all_summaries(results_dir)
    print(f"  Loaded {len(df)} rows from {df['_source_file'].nunique() if '_source_file' in df.columns else '?'} files")

    # ── 1. flat dump ──────────────────────────────────────────────────────
    all_path = out_dir / "all_runs.csv"
    df.drop(columns=["_source_file"], errors="ignore").to_csv(all_path, index=False)
    print(f"Wrote {all_path}")

    # ── 2. aggregate by workload × method × k ────────────────────────────
    agg = agg_group(df, ["workload", "method", "k"])
    agg_path = out_dir / "aggregate_by_workload_method.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Wrote {agg_path}")

    # ── 3. aggregate by run_name × workload × method × k ─────────────────
    if "run_name" in df.columns:
        agg_run = agg_group(df, ["run_name", "workload", "method", "k"])
        agg_run_path = out_dir / "aggregate_by_run.csv"
        agg_run.to_csv(agg_run_path, index=False)
        print(f"Wrote {agg_run_path}")

    # ── 4. quick console summary ──────────────────────────────────────────
    print("\n── Quick summary (workload × method, mean tps / latency) ──")
    if "tps_mean" in agg.columns:
        print(agg[["workload", "method", "k", "tps_mean", "latency_mean",
                    "acceptance_rate_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()