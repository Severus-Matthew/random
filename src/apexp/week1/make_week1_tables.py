#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def slug(x):
    x = "" if pd.isna(x) else str(x)
    x = x.split("/")[-1]
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_") or "NA"


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_cols(df, cols, default=np.nan):
    for c in cols:
        if c not in df.columns:
            df[c] = default
    return df


def add_candidate_id(df):
    ensure_cols(df, [
        "method", "k", "k_actual", "temperature", "draft_model",
        "eagle3_model", "ngram_lookup_min", "ngram_lookup_max",
        "max_prompt_tokens", "num_turns",
    ])

    k_col = "k" if "k" in df.columns else "k_actual"

    def one(r):
        method = str(r.get("method", "NA"))
        k = r.get(k_col, r.get("k_actual", "NA"))
        temp = r.get("temperature", 0.0)
        parts = [method, f"k{k}", f"t{temp}"]

        if method == "draft_sd":
            parts.append("draft_" + slug(r.get("draft_model", "NA")))

        if method == "eagle3":
            parts.append("eagle_" + slug(r.get("eagle3_model", "eagle3")))

        if method == "ngram_sd":
            parts.append(f"ng{r.get('ngram_lookup_min','NA')}-{r.get('ngram_lookup_max','NA')}")

        ctx = r.get("max_prompt_tokens", 0)
        turns = r.get("num_turns", 1)
        try:
            if float(ctx) > 0:
                parts.append(f"ctx{int(float(ctx))}")
        except Exception:
            pass
        try:
            if float(turns) > 1:
                parts.append(f"turns{int(float(turns))}")
        except Exception:
            pass

        return "|".join(parts)

    df["candidate_id"] = df.apply(one, axis=1)
    return df


def filter_core(df):
    out = df.copy()

    if "temperature" in out.columns:
        out = out[pd.to_numeric(out["temperature"], errors="coerce").fillna(0.0).eq(0.0)]

    if "max_prompt_tokens" in out.columns:
        ctx = pd.to_numeric(out["max_prompt_tokens"], errors="coerce").fillna(0)
        out = out[ctx.eq(0)]

    if "num_turns" in out.columns:
        turns = pd.to_numeric(out["num_turns"], errors="coerce").fillna(1)
        out = out[turns.eq(1)]

    return out


def p95(x):
    return np.nanpercentile(x, 95) if len(x) else np.nan


def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "week1"
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    req_path = root / "request_summary_all.csv"
    block_path = root / "block_joined" / "block_with_request_features.csv"
    status_path = root / "job_status.tsv"

    req = pd.read_csv(req_path)
    block = pd.read_csv(block_path)

    req = to_num(req, [
        "k", "temperature", "max_prompt_tokens", "num_turns",
        "latency_s", "tokens_per_sec", "n_output_tokens",
        "mean_entropy", "repetition_density", "acceptance_rate",
        "draft_tokens", "accepted_tokens_total",
    ])
    block = to_num(block, [
        "k", "k_actual", "temperature", "max_prompt_tokens", "num_turns",
        "accepted_len", "first_rejection", "num_rejected",
        "prompt_token_len", "latency_s", "tokens_per_sec",
        "n_output_tokens", "mean_entropy", "repetition_density",
    ])

    req = add_candidate_id(req)
    block = add_candidate_id(block)

    # Save status summary.
    if status_path.exists():
        status = pd.read_csv(status_path, sep="\t")
        latest = status.drop_duplicates("job_id", keep="last")
        latest.to_csv(table_dir / "job_status_latest.csv", index=False)
        latest[~latest["status"].isin(["SUCCESS", "SKIPPED"])].to_csv(
            table_dir / "failed_jobs.csv", index=False
        )

    # Core subset for the main Week-1 controller story.
    req_core = filter_core(req)
    block_core = filter_core(block)

    req.to_csv(table_dir / "request_summary_all_normalized.csv", index=False)
    block.to_csv(table_dir / "block_with_request_features_normalized.csv", index=False)
    req_core.to_csv(table_dir / "request_summary_core_t0.csv", index=False)
    block_core.to_csv(table_dir / "block_core_t0.csv", index=False)

    # Request-level performance.
    perf = req_core.groupby(["workload", "method", "k", "candidate_id"], dropna=False).agg(
        n_requests=("tokens_per_sec", "size"),
        mean_tps=("tokens_per_sec", "mean"),
        median_tps=("tokens_per_sec", "median"),
        std_tps=("tokens_per_sec", "std"),
        mean_latency_s=("latency_s", "mean"),
        p95_latency_s=("latency_s", p95),
        mean_output_tokens=("n_output_tokens", "mean"),
        mean_entropy=("mean_entropy", "mean"),
        mean_repetition_density=("repetition_density", "mean"),
        mean_acceptance_rate=("acceptance_rate", "mean"),
    ).reset_index()

    ar = perf[perf["method"] == "ar"][["workload", "mean_tps", "mean_latency_s"]].rename(
        columns={"mean_tps": "ar_mean_tps", "mean_latency_s": "ar_mean_latency_s"}
    )

    perf = perf.merge(ar, on="workload", how="left")
    perf["speedup_vs_ar"] = perf["mean_tps"] / perf["ar_mean_tps"]
    perf["latency_ratio_vs_ar"] = perf["mean_latency_s"] / perf["ar_mean_latency_s"]
    perf.to_csv(table_dir / "table_request_speedup_by_workload_method_k.csv", index=False)

    best_fixed_by_workload = perf.sort_values(
        ["workload", "speedup_vs_ar", "mean_tps"], ascending=[True, False, False]
    ).groupby("workload").head(1)
    best_fixed_by_workload.to_csv(table_dir / "table_best_fixed_by_workload.csv", index=False)

    best_global = perf[perf["method"] != "ar"].groupby(["candidate_id", "method", "k"], dropna=False).agg(
        mean_speedup_vs_ar=("speedup_vs_ar", "mean"),
        mean_tps=("mean_tps", "mean"),
        n_workloads=("workload", "nunique"),
    ).reset_index().sort_values("mean_speedup_vs_ar", ascending=False)
    best_global.to_csv(table_dir / "table_best_global_fixed.csv", index=False)

    # Block-level accepted length and rejection.
    if "full_accept" in block_core.columns:
        if block_core["full_accept"].dtype == object:
            block_core["full_accept_bool"] = block_core["full_accept"].astype(str).str.lower().eq("true")
        else:
            block_core["full_accept_bool"] = block_core["full_accept"].astype(bool)
    else:
        block_core["full_accept_bool"] = block_core["accepted_len"].eq(block_core["k_actual"])

    block_stats = block_core.groupby(["workload", "method", "k_actual", "candidate_id"], dropna=False).agg(
        n_blocks=("accepted_len", "size"),
        n_prompts=("prompt_hash", "nunique"),
        mean_accepted_len=("accepted_len", "mean"),
        median_accepted_len=("accepted_len", "median"),
        full_accept_rate=("full_accept_bool", "mean"),
        mean_first_rejection=("first_rejection", "mean"),
        mean_num_rejected=("num_rejected", "mean"),
        mean_prompt_token_len=("prompt_token_len", "mean"),
        mean_entropy=("mean_entropy", "mean"),
        mean_repetition_density=("repetition_density", "mean"),
        mean_latency_s=("latency_s", "mean"),
        mean_tps=("tokens_per_sec", "mean"),
    ).reset_index()
    block_stats.to_csv(table_dir / "table_block_acceptance_by_workload_method_k.csv", index=False)

    # Survival curves from accepted_len.
    max_k = int(np.nanmax(block_core["k_actual"])) if len(block_core) else 0
    surv_rows = []
    for keys, g in block_core.groupby(["workload", "method", "k_actual", "candidate_id"], dropna=False):
        workload, method, k_actual, candidate = keys
        k_int = int(k_actual)
        row = {
            "workload": workload,
            "method": method,
            "k_actual": k_actual,
            "candidate_id": candidate,
            "n_blocks": len(g),
        }
        for j in range(k_int):
            at_risk = g["accepted_len"] >= j
            row[f"p_survive_pos_{j}"] = float((g["accepted_len"] > j).mean())
            denom = int(at_risk.sum())
            row[f"hazard_pos_{j}"] = float(((g["accepted_len"] == j) & at_risk).sum() / denom) if denom else np.nan
        surv_rows.append(row)

    surv = pd.DataFrame(surv_rows)
    surv.to_csv(table_dir / "table_block_survival_curves.csv", index=False)

    # Oracle by prompt: best actual candidate per prompt_hash/workload.
    req_oracle_base = req_core.dropna(subset=["prompt_hash", "tokens_per_sec"]).copy()
    group_cols = ["workload", "prompt_hash"]
    candidate_counts = req_oracle_base.groupby(group_cols)["candidate_id"].nunique().reset_index(name="n_candidates")
    valid_groups = candidate_counts[candidate_counts["n_candidates"] >= 2][group_cols]
    req_oracle_base = req_oracle_base.merge(valid_groups, on=group_cols, how="inner")

    if len(req_oracle_base):
        oracle = req_oracle_base.sort_values(
            group_cols + ["tokens_per_sec"], ascending=[True, True, False]
        ).groupby(group_cols).head(1)
        oracle.to_csv(table_dir / "table_oracle_best_candidate_per_prompt.csv", index=False)

        oracle_summary = oracle.groupby(["workload", "candidate_id", "method", "k"], dropna=False).agg(
            chosen_count=("prompt_hash", "size"),
            mean_oracle_tps=("tokens_per_sec", "mean"),
        ).reset_index()
        oracle_summary.to_csv(table_dir / "table_oracle_choice_distribution.csv", index=False)

    # Dataset summary.
    summary = {
        "root": str(root),
        "n_request_rows": int(len(req)),
        "n_block_rows": int(len(block)),
        "n_core_request_rows": int(len(req_core)),
        "n_core_block_rows": int(len(block_core)),
        "n_workloads_request": int(req["workload"].nunique()) if "workload" in req.columns else 0,
        "n_workloads_block": int(block["workload"].nunique()) if "workload" in block.columns else 0,
        "n_candidates_core": int(req_core["candidate_id"].nunique()) if "candidate_id" in req_core.columns else 0,
    }
    with open(table_dir / "week1_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots.
    # 1. Global fixed speedup.
    top = best_global.head(15).iloc[::-1]
    if len(top):
        plt.figure(figsize=(10, 6))
        plt.barh(top["candidate_id"], top["mean_speedup_vs_ar"])
        plt.xlabel("Mean speedup vs AR")
        plt.ylabel("Candidate")
        plt.title("Top fixed candidates across workloads")
        savefig(plot_dir / "plot_top_fixed_candidates_speedup.png")

    # 2. Accepted length vs k.
    acc = block_stats.groupby(["method", "k_actual"], dropna=False)["mean_accepted_len"].mean().reset_index()
    if len(acc):
        plt.figure(figsize=(8, 5))
        for method, g in acc.groupby("method"):
            g = g.sort_values("k_actual")
            plt.plot(g["k_actual"], g["mean_accepted_len"], marker="o", label=method)
        plt.xlabel("k")
        plt.ylabel("Mean accepted length")
        plt.title("Mean block accepted length vs k")
        plt.legend()
        savefig(plot_dir / "plot_mean_accepted_len_vs_k.png")

    # 3. Full accept rate vs k.
    fa = block_stats.groupby(["method", "k_actual"], dropna=False)["full_accept_rate"].mean().reset_index()
    if len(fa):
        plt.figure(figsize=(8, 5))
        for method, g in fa.groupby("method"):
            g = g.sort_values("k_actual")
            plt.plot(g["k_actual"], g["full_accept_rate"], marker="o", label=method)
        plt.xlabel("k")
        plt.ylabel("Full accept rate")
        plt.title("Full block acceptance probability vs k")
        plt.legend()
        savefig(plot_dir / "plot_full_accept_rate_vs_k.png")

    # 4. Oracle choice distribution.
    oracle_path = table_dir / "table_oracle_choice_distribution.csv"
    if oracle_path.exists():
        od = pd.read_csv(oracle_path)
        od2 = od.groupby("candidate_id")["chosen_count"].sum().sort_values(ascending=False).head(15).iloc[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(od2.index.astype(str), od2.values)
        plt.xlabel("Number of prompt groups where candidate is oracle")
        plt.ylabel("Candidate")
        plt.title("Oracle candidate distribution")
        savefig(plot_dir / "plot_oracle_candidate_distribution.png")

    print("Wrote Week-1 tables to:", table_dir)
    print("Wrote Week-1 plots to:", plot_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
