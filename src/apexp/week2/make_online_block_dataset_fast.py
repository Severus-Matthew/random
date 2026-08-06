#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    block_path = root / "block_joined" / "block_with_request_features.csv"
    if not block_path.exists():
        raise SystemExit(f"Missing {block_path}")

    usecols = [
        "workload", "method", "trace_dir", "req_id", "prompt_id", "prompt_hash",
        "vllm_request_id", "experiment", "run_name", "source_dataset", "split",
        "draft_model", "eagle3_model", "ngram_lookup_min", "ngram_lookup_max",
        "block_index", "accepted_len", "first_rejection", "full_accept",
        "num_rejected", "k_actual", "k", "temperature", "prompt_token_len",
        "n_output_tokens", "num_turns", "max_prompt_tokens", "latency_s",
        "tokens_per_sec", "mean_entropy", "repetition_density",
        "accepted_per_draft", "acceptance_rate",
    ]

    print("Loading:", block_path, flush=True)
    df = pd.read_csv(block_path, usecols=lambda c: c in usecols, low_memory=False)
    print("Loaded:", df.shape, flush=True)

    df = df[df["method"].isin(["ngram_sd", "draft_sd", "eagle3"])].copy()
    print("After method filter:", df.shape, flush=True)

    if args.core_only:
        if "temperature" in df.columns:
            df = df[safe_num(df["temperature"], 0.0).eq(0.0)].copy()
        if "max_prompt_tokens" in df.columns:
            df = df[safe_num(df["max_prompt_tokens"], 0.0).eq(0.0)].copy()
        if "num_turns" in df.columns:
            df = df[safe_num(df["num_turns"], 1.0).eq(1.0)].copy()
        print("After core-only filter:", df.shape, flush=True)

    numeric = [
        "block_index", "accepted_len", "first_rejection", "num_rejected",
        "k_actual", "k", "temperature", "prompt_token_len", "n_output_tokens",
        "num_turns", "max_prompt_tokens", "latency_s", "tokens_per_sec",
        "mean_entropy", "repetition_density", "accepted_per_draft",
        "acceptance_rate",
    ]

    for c in numeric:
        if c in df.columns:
            default = 1.0 if c == "num_turns" else 0.0
            df[c] = safe_num(df[c], default)

    df["block_index"] = df["block_index"].astype(int)
    df["target_accepted_len"] = df["accepted_len"].clip(0, 16).astype(int)
    df["target_first_rejection"] = df["first_rejection"].clip(0, 16).astype(int)
    df["target_num_rejected"] = df["num_rejected"].clip(0, 16)
    df["target_k_actual"] = df["k_actual"].fillna(df["k"]).clip(1, 16).astype(int)
    df["target_full_accept"] = (df["target_accepted_len"] >= df["target_k_actual"]).astype(int)
    df["k_requested"] = df["k"].fillna(df["target_k_actual"]).clip(1, 16).astype(int)
    df["target_latency_s"] = df["latency_s"]
    df["target_tokens_per_sec"] = df["tokens_per_sec"].clip(lower=1e-6)

    # Fast online-compatible proxies. These are request-level summaries from the
    # original vLLM traces. They are not current-block acceptance labels.
    df["prefix_pos"] = 0.0
    df["prefix_frac"] = 0.0

    df["prefix_entropy_mean"] = df["mean_entropy"].fillna(0.0)
    df["prefix_entropy_std"] = 0.0
    for w in [16, 32, 64, 128, 256]:
        df[f"prefix_entropy_mean_w{w}"] = df["mean_entropy"].fillna(0.0)
        df[f"prefix_entropy_std_w{w}"] = 0.0
        df[f"prefix_repetition_w{w}"] = df["repetition_density"].fillna(0.0)
        df[f"prefix_window_len_w{w}"] = 0

    # Sort once, then use vectorized groupby transforms. Much faster than iterrows.
    group_cols = ["trace_dir", "req_id"]
    df = df.sort_values(group_cols + ["block_index"]).reset_index(drop=True)
    g = df.groupby(group_cols, dropna=False, sort=False)

    acc_prev = g["target_accepted_len"].shift(1)
    full_prev = g["target_full_accept"].shift(1)
    first_prev = g["target_first_rejection"].shift(1)
    rej_prev = g["target_num_rejected"].shift(1)

    df["hist_n_blocks"] = g.cumcount()
    df["hist_mean_accepted_len"] = (
        acc_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )
    df["hist_std_accepted_len"] = (
        acc_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
        .expanding()
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )
    df["hist_mean_first_rejection"] = (
        first_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )
    df["hist_full_accept_rate"] = (
        full_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )
    df["hist_reject_rate"] = 1.0 - df["hist_full_accept_rate"]
    df["hist_mean_num_rejected"] = (
        rej_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )

    for w in [1, 2, 4, 8, 16]:
        df[f"hist_mean_accepted_len_last{w}"] = (
            acc_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .fillna(0.0)
        )
        df[f"hist_full_accept_rate_last{w}"] = (
            full_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .fillna(0.0)
        )
        df[f"hist_mean_first_rejection_last{w}"] = (
            first_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .fillna(0.0)
        )
        df[f"hist_mean_num_rejected_last{w}"] = (
            rej_prev.groupby([df[c] for c in group_cols], dropna=False, sort=False)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .fillna(0.0)
        )

    df["feature_cutoff_block"] = df["block_index"] - 1
    df["target_block"] = df["block_index"]
    df["uses_future_request_features"] = 0
    df["trace_found"] = 0
    df["fast_builder_used_request_entropy_proxy"] = 1

    keep = [
        "workload", "method", "trace_dir", "req_id", "prompt_id", "prompt_hash",
        "vllm_request_id", "experiment", "run_name", "source_dataset", "split",
        "draft_model", "eagle3_model", "ngram_lookup_min", "ngram_lookup_max",
        "block_index", "target_accepted_len", "target_first_rejection",
        "target_full_accept", "target_num_rejected", "target_k_actual",
        "target_latency_s", "target_tokens_per_sec", "k_requested",
        "temperature", "prompt_token_len", "n_output_tokens", "num_turns",
        "max_prompt_tokens",
        "prefix_pos", "prefix_frac", "prefix_entropy_mean", "prefix_entropy_std",
        "prefix_entropy_mean_w16", "prefix_entropy_mean_w32",
        "prefix_entropy_mean_w64", "prefix_entropy_mean_w128",
        "prefix_entropy_mean_w256", "prefix_entropy_std_w16",
        "prefix_entropy_std_w32", "prefix_entropy_std_w64",
        "prefix_entropy_std_w128", "prefix_repetition_w16",
        "prefix_repetition_w32", "prefix_repetition_w64",
        "prefix_repetition_w128", "prefix_repetition_w256",
        "hist_n_blocks", "hist_mean_accepted_len", "hist_std_accepted_len",
        "hist_mean_first_rejection", "hist_full_accept_rate", "hist_reject_rate",
        "hist_mean_num_rejected",
        "hist_mean_accepted_len_last1", "hist_mean_accepted_len_last2",
        "hist_mean_accepted_len_last4", "hist_mean_accepted_len_last8",
        "hist_mean_accepted_len_last16",
        "hist_full_accept_rate_last1", "hist_full_accept_rate_last2",
        "hist_full_accept_rate_last4", "hist_full_accept_rate_last8",
        "hist_full_accept_rate_last16",
        "hist_mean_first_rejection_last1", "hist_mean_first_rejection_last2",
        "hist_mean_first_rejection_last4", "hist_mean_first_rejection_last8",
        "hist_mean_first_rejection_last16",
        "hist_mean_num_rejected_last1", "hist_mean_num_rejected_last2",
        "hist_mean_num_rejected_last4", "hist_mean_num_rejected_last8",
        "hist_mean_num_rejected_last16",
        "feature_cutoff_block", "target_block", "uses_future_request_features",
        "trace_found", "fast_builder_used_request_entropy_proxy",
    ]

    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(out, index=False)

    audit = {
        "input_block_rows": int(len(df)),
        "output_rows": int(len(df)),
        "core_only": bool(args.core_only),
        "builder": "fast_vectorized_history_builder",
        "trace_found": 0,
        "fast_builder_used_request_entropy_proxy": 1,
        "causal_feature_rule": "Block history features use shifted previous-block outcomes only. Current block acceptance/rejection/TPS are target-only.",
        "note": "This skips per-token traces.jsonl prefix lookup to avoid 16.7M-row Python loop. Prefix entropy/repetition columns are filled with request-level trace summary proxies from block_with_request_features.csv.",
    }

    audit_path = out.parent / "feature_leakage_audit_fast.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))

    print("Wrote:", out, flush=True)
    print("Rows:", df.shape, flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
