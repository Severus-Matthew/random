#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "workload",
    "method",
    "trace_dir",
    "req_id",
    "prompt_hash",
    "block_index",
    "target_accepted_len",
    "target_first_rejection",
    "target_full_accept",
    "target_num_rejected",
    "target_k_actual",
    "target_tokens_per_sec",
    "k_requested",
    "prompt_token_len",
    "prefix_pos",
    "prefix_frac",
    "hist_n_blocks",
    "hist_mean_accepted_len",
    "hist_std_accepted_len",
    "hist_mean_first_rejection",
    "hist_full_accept_rate",
    "hist_reject_rate",
    "hist_mean_num_rejected",
    "hist_mean_accepted_len_last1",
    "hist_mean_accepted_len_last2",
    "hist_mean_accepted_len_last4",
    "hist_mean_accepted_len_last8",
    "hist_mean_accepted_len_last16",
    "hist_full_accept_rate_last1",
    "hist_full_accept_rate_last2",
    "hist_full_accept_rate_last4",
    "hist_full_accept_rate_last8",
    "hist_full_accept_rate_last16",
    "hist_mean_first_rejection_last4",
    "hist_mean_num_rejected_last4",
    "feature_cutoff_block",
    "target_block",
    "uses_future_request_features",
    "trace_found",
]


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def fail(msg):
    raise RuntimeError(msg)


def approx(a, b, tol=1e-6):
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(float(a) - float(b)) <= tol


def validate_causal_history_sample(df: pd.DataFrame, n_groups: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    group_cols = ["trace_dir", "req_id"]

    keys = df[group_cols].drop_duplicates()
    if len(keys) == 0:
        return ["No groups found for causal validation."]

    take = min(n_groups, len(keys))
    chosen_idx = rng.choice(len(keys), size=take, replace=False)
    chosen = keys.iloc[chosen_idx]

    problems = []

    indexed = df.set_index(group_cols, drop=False)

    for _, keyrow in chosen.iterrows():
        trace_dir = keyrow["trace_dir"]
        req_id = keyrow["req_id"]

        try:
            g = indexed.loc[(trace_dir, req_id)].copy()
        except Exception:
            continue

        if isinstance(g, pd.Series):
            g = g.to_frame().T

        g = g.sort_values("block_index").reset_index(drop=True)

        past_acc = []
        past_full = []
        past_first = []
        past_rej = []

        for i, r in g.iterrows():
            b = int(r["block_index"])

            checks = {
                "hist_n_blocks": len(past_acc),
                "hist_mean_accepted_len": float(np.mean(past_acc)) if past_acc else 0.0,
                "hist_std_accepted_len": float(np.std(past_acc)) if past_acc else 0.0,
                "hist_mean_first_rejection": float(np.mean(past_first)) if past_first else 0.0,
                "hist_full_accept_rate": float(np.mean(past_full)) if past_full else 0.0,
                "hist_reject_rate": 1.0 - float(np.mean(past_full)) if past_full else 0.0,
                "hist_mean_num_rejected": float(np.mean(past_rej)) if past_rej else 0.0,
            }

            for w in [1, 2, 4, 8, 16]:
                pa = past_acc[-w:]
                pf = past_full[-w:]
                pr = past_first[-w:]
                nr = past_rej[-w:]

                checks[f"hist_mean_accepted_len_last{w}"] = float(np.mean(pa)) if pa else 0.0
                checks[f"hist_full_accept_rate_last{w}"] = float(np.mean(pf)) if pf else 0.0
                checks[f"hist_mean_first_rejection_last{w}"] = float(np.mean(pr)) if pr else 0.0
                checks[f"hist_mean_num_rejected_last{w}"] = float(np.mean(nr)) if nr else 0.0

            for col, expected in checks.items():
                got = r.get(col)
                if not approx(got, expected, tol=1e-5):
                    problems.append(
                        f"group=({trace_dir},{req_id}) block={b} col={col} got={got} expected={expected}"
                    )
                    if len(problems) >= 30:
                        return problems

            acc = int(r["target_accepted_len"])
            k = int(r["target_k_actual"])
            first = int(r["target_first_rejection"])
            rej = int(r["target_num_rejected"])
            full = int(r["target_full_accept"])

            past_acc.append(acc)
            past_full.append(full)
            past_first.append(first)
            past_rej.append(rej)

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--block-source", required=True)
    ap.add_argument("--audit", default="")
    ap.add_argument("--sample-groups", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    block_path = Path(args.block_source)

    if not data_path.exists():
        fail(f"Missing dataset: {data_path}")
    if not block_path.exists():
        fail(f"Missing block source: {block_path}")

    print("[read header]", data_path)
    header = pd.read_csv(data_path, nrows=0)
    cols = list(header.columns)
    print("n_cols:", len(cols))

    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        fail(f"Missing required columns: {missing}")

    forbidden = [
        "mean_entropy",
        "repetition_density",
        "entropy_bucket",
        "oracle_k_estimated",
        "oracle_expected_tokens",
    ]
    present_forbidden = [c for c in forbidden if c in cols]
    if present_forbidden:
        fail(f"Forbidden/leaky request aggregate columns present: {present_forbidden}")

    print("[load online dataset]")
    df = pd.read_csv(data_path, low_memory=False)
    print("shape:", df.shape)

    print("[load source block dataset core-filtered count]")
    usecols = ["method", "temperature", "max_prompt_tokens", "num_turns"]
    src = pd.read_csv(block_path, usecols=lambda c: c in usecols, low_memory=False)

    src = src[src["method"].isin(["ngram_sd", "draft_sd", "eagle3"])].copy()
    if "temperature" in src.columns:
        src = src[pd.to_numeric(src["temperature"], errors="coerce").fillna(0.0).eq(0.0)]
    if "max_prompt_tokens" in src.columns:
        src = src[pd.to_numeric(src["max_prompt_tokens"], errors="coerce").fillna(0).eq(0)]
    if "num_turns" in src.columns:
        src = src[pd.to_numeric(src["num_turns"], errors="coerce").fillna(1).eq(1)]

    expected_rows = len(src)
    print("expected core rows:", expected_rows)
    print("online rows:", len(df))

    if len(df) != expected_rows:
        fail(f"Row-count mismatch. online={len(df)} expected_core={expected_rows}")

    numeric_cols = [
        "block_index",
        "target_accepted_len",
        "target_first_rejection",
        "target_full_accept",
        "target_num_rejected",
        "target_k_actual",
        "target_tokens_per_sec",
        "k_requested",
        "prompt_token_len",
        "prefix_pos",
        "prefix_frac",
        "hist_n_blocks",
        "hist_mean_accepted_len",
        "hist_std_accepted_len",
        "hist_mean_first_rejection",
        "hist_full_accept_rate",
        "hist_reject_rate",
        "hist_mean_num_rejected",
        "feature_cutoff_block",
        "target_block",
        "uses_future_request_features",
        "trace_found",
    ]

    print("[numeric NaN/inf check]")
    bad_numeric = {}
    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        n_bad = int((~np.isfinite(x.to_numpy(dtype=float))).sum())
        if n_bad:
            bad_numeric[c] = n_bad

    if bad_numeric:
        fail(f"Bad numeric values found: {bad_numeric}")

    print("[range checks]")
    checks = []

    checks.append(("accepted_len range", df["target_accepted_len"].between(0, 16).all()))
    checks.append(("first_rejection range", df["target_first_rejection"].between(0, 16).all()))
    checks.append(("num_rejected range", df["target_num_rejected"].between(0, 16).all()))
    checks.append(("target_k_actual range", df["target_k_actual"].between(1, 16).all()))
    checks.append(("k_requested values", set(pd.to_numeric(df["k_requested"]).dropna().astype(int).unique()).issubset({1, 2, 4, 8, 16})))
    checks.append(("accepted_len <= k", (df["target_accepted_len"] <= df["target_k_actual"]).all()))
    checks.append(("first_rejection <= k", (df["target_first_rejection"] <= df["target_k_actual"]).all()))
    checks.append(("full_accept binary", set(pd.to_numeric(df["target_full_accept"]).dropna().astype(int).unique()).issubset({0, 1})))
    checks.append(("uses_future_request_features zero", (pd.to_numeric(df["uses_future_request_features"]) == 0).all()))
    checks.append(("feature_cutoff_block = target_block - 1", (pd.to_numeric(df["feature_cutoff_block"]) == pd.to_numeric(df["target_block"]) - 1).all()))
    checks.append(("prefix_frac range", pd.to_numeric(df["prefix_frac"]).between(0, 1).all()))
    checks.append(("target_tps positive", (pd.to_numeric(df["target_tokens_per_sec"]) > 0).all()))

    failed = [name for name, ok in checks if not bool(ok)]
    if failed:
        fail(f"Range/invariant checks failed: {failed}")

    print("[distribution summary]")
    summary = {
        "shape": list(df.shape),
        "methods": df["method"].value_counts(dropna=False).to_dict(),
        "workloads": df["workload"].value_counts(dropna=False).to_dict(),
        "k_requested": pd.to_numeric(df["k_requested"]).value_counts(dropna=False).sort_index().to_dict(),
        "target_k_actual": pd.to_numeric(df["target_k_actual"]).value_counts(dropna=False).sort_index().to_dict(),
        "trace_found": pd.to_numeric(df["trace_found"]).value_counts(dropna=False).sort_index().to_dict(),
        "accepted_len_hist": pd.to_numeric(df["target_accepted_len"]).value_counts(dropna=False).sort_index().to_dict(),
        "target_tps_quantiles": pd.to_numeric(df["target_tokens_per_sec"]).quantile([0, .01, .05, .5, .95, .99, 1]).to_dict(),
        "prompt_token_len_quantiles": pd.to_numeric(df["prompt_token_len"]).quantile([0, .01, .05, .5, .95, .99, 1]).to_dict(),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    print("[causal history validation sample]")
    problems = validate_causal_history_sample(df, n_groups=args.sample_groups, seed=args.seed)
    if problems:
        print("First causal-history problems:")
        for p in problems[:30]:
            print(p)
        fail(f"Causal history validation failed with {len(problems)} sampled problems")

    if args.audit:
        audit_path = Path(args.audit)
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            print("[audit file]")
            print(json.dumps(audit, indent=2, sort_keys=True))
            if int(audit.get("output_rows", len(df))) != len(df):
                fail("Audit output_rows does not match actual dataset rows")
        else:
            print(f"[warn] audit file not found: {audit_path}")

    out_report = data_path.parent / "online_block_dataset_core_validation_report.json"
    out_report.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("[PASS] dataset looks valid")
    print("wrote:", out_report)


if __name__ == "__main__":
    main()
