#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_all_summary_csv(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("summary.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
            continue
        if df.empty:
            continue
        df["summary_path"] = str(p)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "k", "tokens_per_sec", "latency_s", "n_output_tokens",
        "acceptance_rate", "draft_tokens", "accepted_tokens_total",
        "rejected_tokens", "rejection_rate", "accepted_tokens_per_verifier_pass",
        "mean_entropy", "prompt_token_len",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = add_numeric(df)

    for c in ["draft_tokens", "accepted_tokens_total", "rejected_tokens"]:
        if c not in d.columns:
            d[c] = np.nan

    out = (
        d.groupby(group_cols, dropna=False)
        .agg(
            n=("tokens_per_sec", "size"),
            mean_tps=("tokens_per_sec", "mean"),
            median_tps=("tokens_per_sec", "median"),
            mean_latency_s=("latency_s", "mean"),
            median_latency_s=("latency_s", "median"),
            mean_output_tokens=("n_output_tokens", "mean"),
            mean_acceptance_rate=("acceptance_rate", "mean"),
            mean_rejection_rate=("rejection_rate", "mean"),
            mean_entropy=("mean_entropy", "mean"),
            total_draft_tokens=("draft_tokens", "sum"),
            total_accepted_tokens=("accepted_tokens_total", "sum"),
            total_rejected_tokens=("rejected_tokens", "sum"),
        )
        .reset_index()
    )

    out["wasted_token_rate"] = out["total_rejected_tokens"] / out["total_draft_tokens"].replace(0, np.nan)
    out["accepted_token_rate"] = out["total_accepted_tokens"] / out["total_draft_tokens"].replace(0, np.nan)
    out["wasted_token_pct"] = 100.0 * out["wasted_token_rate"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_all_summary_csv(root)
    if rows.empty:
        raise SystemExit(f"No summary.csv files found under {root}")

    rows = add_numeric(rows)
    rows.to_csv(out_dir / "fixed_baseline_request_rows.csv", index=False)

    by_workload = summarize_group(rows, ["method", "k", "workload"])
    by_workload.to_csv(out_dir / "fixed_baseline_by_workload_micro.csv", index=False)

    overall_micro = summarize_group(rows, ["method", "k"])
    overall_micro.to_csv(out_dir / "fixed_baseline_overall_micro.csv", index=False)

    macro = (
        by_workload.groupby(["method", "k"], dropna=False)
        .agg(
            n_workloads=("workload", "nunique"),
            macro_mean_tps=("mean_tps", "mean"),
            macro_median_tps=("median_tps", "mean"),
            macro_mean_latency_s=("mean_latency_s", "mean"),
            macro_acceptance_rate=("mean_acceptance_rate", "mean"),
            macro_rejection_rate=("mean_rejection_rate", "mean"),
            macro_wasted_token_rate=("wasted_token_rate", "mean"),
            macro_wasted_token_pct=("wasted_token_pct", "mean"),
        )
        .reset_index()
        .sort_values("macro_mean_tps", ascending=False)
    )
    macro.to_csv(out_dir / "fixed_baseline_overall_macro_by_workload.csv", index=False)

    print("WROTE:", out_dir)
    print("request rows:", len(rows))
    print("by workload rows:", len(by_workload))
    print("overall rows:", len(overall_micro))


if __name__ == "__main__":
    main()
