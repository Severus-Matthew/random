#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.apexp.week2.neural_controller_lib import (
    load_model,
    score_dataframe,
    safe_num,
    prompt_group_key,
)


def merge_prompt_text(df, prompt_text_map_path):
    mp = pd.read_csv(prompt_text_map_path)
    df = df.copy()
    df["prompt_hash"] = df["prompt_hash"].astype(str)
    mp["prompt_hash"] = mp["prompt_hash"].astype(str)

    before = len(df)
    df = df.merge(mp[["prompt_hash", "prompt_text"]], on="prompt_hash", how="left")
    assert len(df) == before

    match_rate = float(df["prompt_text"].notna().mean())
    print(f"prompt_text row match rate: {match_rate:.4f}")
    if match_rate < 0.90:
        raise RuntimeError(f"prompt_text row match rate too low: {match_rate:.4f}")

    df["prompt_text"] = df["prompt_text"].fillna("")
    return df


def make_candidate_table(df):
    df = df.copy()
    df["group_key"] = prompt_group_key(df)
    df["k_requested"] = safe_num(df["k_requested"]).fillna(1).astype(int)

    if "target_tokens_per_sec" in df.columns:
        df["actual_tps"] = safe_num(df["target_tokens_per_sec"])
    elif "tokens_per_sec" in df.columns:
        df["actual_tps"] = safe_num(df["tokens_per_sec"])
    else:
        raise ValueError("No target_tokens_per_sec or tokens_per_sec column found.")

    df["actual_tps"] = df["actual_tps"].fillna(0.0)

    sort_cols = []
    if "block_index" in df.columns:
        sort_cols.append("block_index")
    if "prefix_pos" in df.columns:
        sort_cols.append("prefix_pos")
    if sort_cols:
        df = df.sort_values(sort_cols)

    keys = ["group_key", "method", "k_requested"]

    keep_first_cols = [
        "group_key", "prompt_hash", "prompt_text", "method", "k_requested",
        "draft_model", "eagle3_model", "ngram_lookup_min", "ngram_lookup_max",
        "temperature", "prompt_token_len", "prefix_pos", "prefix_frac",
        "workload",
    ]
    keep_first_cols = [c for c in keep_first_cols if c in df.columns]

    first = df.groupby(keys, as_index=False).first()
    actual = (
        df.groupby(keys, as_index=False)["actual_tps"]
        .mean()
        .rename(columns={"actual_tps": "actual_tps_mean"})
    )

    out = first.merge(actual, on=keys, how="left")
    out["actual_tps"] = out["actual_tps_mean"]
    return out


def choose_by_score(cand, score_col):
    idx = cand.groupby("group_key")[score_col].idxmax()
    return cand.loc[idx].copy()

def choose_k_conditioned_on_slow(cand, slow_router_dir, score_col):
    """
    Correct fast-controller evaluator.

    Slow router chooses method.
    Fast controller chooses only k within that slow-selected method.
    """
    slow_path = Path(slow_router_dir) / "test_candidates_scored.csv"
    slow = pd.read_csv(slow_path)

    if "pred_tps" not in slow.columns:
        raise ValueError(f"{slow_path} does not contain pred_tps")

    slow = slow.copy()
    slow["prompt_hash"] = slow["prompt_hash"].astype(str)

    slow_idx = slow.groupby("prompt_hash")["pred_tps"].idxmax()
    slow_selected = slow.loc[
        slow_idx,
        ["prompt_hash", "method", "k_requested"],
    ].copy()

    slow_selected = slow_selected.rename(
        columns={
            "method": "slow_method",
            "k_requested": "slow_k",
        }
    )

    cand = cand.copy()
    cand["prompt_hash"] = cand["prompt_hash"].astype(str)

    before_groups = cand["group_key"].nunique()

    cand = cand.merge(
        slow_selected,
        on="prompt_hash",
        how="inner",
    )

    after_groups = cand["group_key"].nunique()
    if after_groups < 0.90 * before_groups:
        raise RuntimeError(
            f"Lost too many groups when merging slow router choices: "
            f"{after_groups}/{before_groups}"
        )

    fast_cand = cand[cand["method"] == cand["slow_method"]].copy()

    if fast_cand.empty:
        raise RuntimeError("No fast candidates left after conditioning on slow-selected method.")

    idx = fast_cand.groupby("group_key")[score_col].idxmax()
    return fast_cand.loc[idx].copy()


def choose_k_conditioned_on_slow_with_prior_guard(
    cand,
    cand_train,
    slow_router_dir,
    score_col,
    prior_floor=0.95,
    prior_beta=1.0,
):
    """
    Safer corrected fast-controller evaluator.

    Slow router chooses method.
    Fast controller chooses k only within that method.
    A train-only action prior prevents selecting globally dominated k values.
    """
    slow_path = Path(slow_router_dir) / "test_candidates_scored.csv"
    slow = pd.read_csv(slow_path)

    if "pred_tps" not in slow.columns:
        raise ValueError(f"{slow_path} does not contain pred_tps")

    slow = slow.copy()
    slow["prompt_hash"] = slow["prompt_hash"].astype(str)

    slow_idx = slow.groupby("prompt_hash")["pred_tps"].idxmax()
    slow_selected = slow.loc[
        slow_idx,
        ["prompt_hash", "method", "k_requested"],
    ].copy()

    slow_selected = slow_selected.rename(
        columns={
            "method": "slow_method",
            "k_requested": "slow_k",
        }
    )

    cand = cand.copy()
    cand["prompt_hash"] = cand["prompt_hash"].astype(str)

    cand = cand.merge(
        slow_selected,
        on="prompt_hash",
        how="inner",
    )

    fast_cand = cand[cand["method"] == cand["slow_method"]].copy()

    if fast_cand.empty:
        raise RuntimeError("No fast candidates left after conditioning on slow-selected method.")

    # Train-only prior over actions.
    prior = (
        cand_train.groupby(["method", "k_requested"])["actual_tps"]
        .mean()
        .reset_index()
        .rename(columns={"actual_tps": "train_prior_tps"})
    )

    fast_cand = fast_cand.merge(
        prior,
        on=["method", "k_requested"],
        how="left",
    )

    slow_prior = prior.rename(
        columns={
            "method": "slow_method",
            "k_requested": "slow_k",
            "train_prior_tps": "slow_train_prior_tps",
        }
    )

    fast_cand = fast_cand.merge(
        slow_prior,
        on=["slow_method", "slow_k"],
        how="left",
    )

    # Allow candidate k only if its train prior is close to the slow action's prior.
    # Always allow the original slow action as fallback.
    is_slow_action = fast_cand["k_requested"].astype(int) == fast_cand["slow_k"].astype(int)

    fast_cand["allowed_by_prior"] = (
        fast_cand["train_prior_tps"].fillna(-1.0)
        >= prior_floor * fast_cand["slow_train_prior_tps"].fillna(1e9)
    ) | is_slow_action

    safe = fast_cand[fast_cand["allowed_by_prior"]].copy()

    if safe.empty:
        raise RuntimeError("No safe fast candidates left after prior guard.")

    # Combine fast model score with train-only prior.
    safe["fast_rank"] = safe.groupby("group_key")[score_col].rank(pct=True)
    safe["prior_rank"] = safe.groupby("group_key")["train_prior_tps"].rank(pct=True)
    safe["safe_score"] = safe["fast_rank"] + prior_beta * safe["prior_rank"]

    idx = safe.groupby("group_key")["safe_score"].idxmax()
    return safe.loc[idx].copy()

def choose_fixed_action(cand, method, k):
    return cand[(cand["method"] == method) & (cand["k_requested"] == k)].copy()


def train_best_global(cand_train):
    stats = (
        cand_train.groupby(["method", "k_requested"])["actual_tps"]
        .mean()
        .reset_index()
        .sort_values("actual_tps", ascending=False)
    )
    return stats.iloc[0]["method"], int(stats.iloc[0]["k_requested"]), stats


def train_best_per_workload(cand_train):
    if "workload" not in cand_train.columns:
        return pd.DataFrame(), pd.DataFrame()

    stats = (
        cand_train.groupby(["workload", "method", "k_requested"])["actual_tps"]
        .mean()
        .reset_index()
        .sort_values(["workload", "actual_tps"], ascending=[True, False])
    )
    best = stats.groupby("workload", as_index=False).first()
    return best, stats


def choose_best_per_workload(cand_test, best):
    rows = []
    for _, r in best.iterrows():
        rows.append(cand_test[
            (cand_test["workload"] == r["workload"]) &
            (cand_test["method"] == r["method"]) &
            (cand_test["k_requested"] == int(r["k_requested"]))
        ])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def choose_oracle(cand):
    idx = cand.groupby("group_key")["actual_tps"].idxmax()
    return cand.loc[idx].copy()


def summarize_policy(name, selected, oracle, ar_tps):
    if selected.empty:
        return {
            "policy": name,
            "n_requests": 0,
            "mean_tps": np.nan,
            "speedup_vs_ar": np.nan,
            "oracle_fraction": np.nan,
        }

    merged = selected[["group_key", "actual_tps"]].merge(
        oracle[["group_key", "actual_tps"]].rename(columns={"actual_tps": "oracle_tps"}),
        on="group_key",
        how="inner",
    )

    mean_tps = float(merged["actual_tps"].mean())
    oracle_mean = float(merged["oracle_tps"].mean())

    return {
        "policy": name,
        "n_requests": int(len(merged)),
        "mean_tps": mean_tps,
        "speedup_vs_ar": mean_tps / ar_tps if ar_tps else np.nan,
        "oracle_fraction": mean_tps / oracle_mean if oracle_mean > 0 else np.nan,
    }


def score_with_model(cand, model_dir, name, variant):
    model, pre, metadata, embedder, device = load_model(model_dir)
    if metadata.get("uses_workload_as_input", False):
        raise RuntimeError(f"{model_dir} uses workload as input; not allowed for deployable replay.")

    scores = score_dataframe(
        model,
        pre,
        cand,
        device=device,
        embedder=embedder,
        variant=variant,
    )

    cand[f"{name}_score"] = scores["pred_score"].to_numpy()
    cand[f"{name}_pred_eacc"] = scores["pred_expected_accepted_len"].to_numpy()
    cand[f"{name}_pred_log_cost"] = scores["pred_log_cost"].to_numpy()
    cand[f"{name}_pred_log_tps"] = scores["pred_log_tps"].to_numpy()
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompt-text-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--full-model", required=True)
    ap.add_argument("--no-cost-model")
    ap.add_argument("--direct-model")
    ap.add_argument("--slow-router-dir", required=True)
    ap.add_argument("--prior-floor", type=float, default=0.95)
    ap.add_argument("--prior-beta", type=float, default=1.0)
    ap.add_argument("--ar-tps", type=float, default=63.726752)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    df = df[df["method"].isin(["ngram_sd", "draft_sd", "eagle3"])].copy()
    df = merge_prompt_text(df, args.prompt_text_map)

    cand = make_candidate_table(df)

    split_path = Path(args.full_model) / "split_groups.json"
    split = json.loads(split_path.read_text())
    train_groups = set(split["train_prompt_hashes"])
    test_groups = set(split["test_prompt_hashes"])

    cand["group_key"] = cand["group_key"].astype(str)
    cand_train = cand[cand["group_key"].isin(train_groups)].copy()
    cand_test = cand[cand["group_key"].isin(test_groups)].copy()

    print("candidate train:", cand_train.shape)
    print("candidate test:", cand_test.shape)

    cand_test = score_with_model(cand_test, args.full_model, "apexp_full", "full")
    if args.no_cost_model:
        cand_test = score_with_model(cand_test, args.no_cost_model, "apexp_no_cost", "no_cost")
    if args.direct_model:
        cand_test = score_with_model(cand_test, args.direct_model, "apexp_direct", "direct")

    oracle = choose_oracle(cand_test)

    policies = []
    policies.append(("oracle", oracle))

    for method, k, name in [
        ("eagle3", 4, "fixed_eagle3_k4"),
        ("ngram_sd", 16, "fixed_ngram_k16"),
        ("draft_sd", 4, "fixed_draft_k4"),
    ]:
        policies.append((name, choose_fixed_action(cand_test, method, k)))

    m, k, global_stats = train_best_global(cand_train)
    global_stats.to_csv(out_dir / "train_global_action_stats.csv", index=False)
    policies.append((f"best_global_fixed_{m}_k{k}", choose_fixed_action(cand_test, m, k)))

    best_w, workload_stats = train_best_per_workload(cand_train)
    if not best_w.empty:
        best_w.to_csv(out_dir / "train_best_per_workload_actions.csv", index=False)
        workload_stats.to_csv(out_dir / "train_workload_action_stats.csv", index=False)
        policies.append(("best_per_workload_fixed_reporting_only", choose_best_per_workload(cand_test, best_w)))

    # Correct fast-controller replay:
    # slow router chooses method, fast controller chooses k only within that method.
    policies.append((
        "apexp_full_konly_conditioned_on_slow",
        choose_k_conditioned_on_slow(
            cand_test,
            args.slow_router_dir,
            "apexp_full_score",
        ),
    ))

    policies.append((
        f"apexp_full_konly_safe_prior_floor_{args.prior_floor}",
        choose_k_conditioned_on_slow_with_prior_guard(
            cand_test,
            cand_train,
            args.slow_router_dir,
            "apexp_full_score",
            prior_floor=args.prior_floor,
            prior_beta=args.prior_beta,
        ),
    ))

    if args.no_cost_model:
        policies.append((
            "apexp_no_cost_konly_conditioned_on_slow",
            choose_k_conditioned_on_slow(
                cand_test,
                args.slow_router_dir,
                "apexp_no_cost_score",
            ),
        ))

    if args.direct_model:
        policies.append((
            "apexp_direct_konly_conditioned_on_slow",
            choose_k_conditioned_on_slow(
                cand_test,
                args.slow_router_dir,
                "apexp_direct_score",
            ),
        ))
    summary = []
    for name, sel in policies:
        summary.append(summarize_policy(name, sel, oracle, args.ar_tps))
        if not sel.empty:
            sel.to_csv(out_dir / f"selected_{name}.csv", index=False)

    summary_df = pd.DataFrame(summary).sort_values("mean_tps", ascending=False)
    summary_df.to_csv(out_dir / "policy_summary.csv", index=False)

    full_sel_konly = choose_k_conditioned_on_slow(
        cand_test,
        args.slow_router_dir,
        "apexp_full_score",
    )

    dist = (
        full_sel_konly.groupby(["method", "k_requested"])
        .size()
        .reset_index(name="count")
    )
    dist["fraction"] = dist["count"] / dist["count"].sum()
    dist = dist.sort_values("fraction", ascending=False)
    dist.to_csv(out_dir / "apexp_full_konly_action_distribution.csv", index=False)

    full_sel_safe = choose_k_conditioned_on_slow_with_prior_guard(
        cand_test,
        cand_train,
        args.slow_router_dir,
        "apexp_full_score",
        prior_floor=args.prior_floor,
        prior_beta=args.prior_beta,
    )

    safe_dist = (
        full_sel_safe.groupby(["method", "k_requested"])
        .size()
        .reset_index(name="count")
    )
    safe_dist["fraction"] = safe_dist["count"] / safe_dist["count"].sum()
    safe_dist = safe_dist.sort_values("fraction", ascending=False)
    safe_dist.to_csv(out_dir / "apexp_full_konly_safe_action_distribution.csv", index=False)
    cand_test.to_csv(out_dir / "candidate_table_test_scored.csv", index=False)

    print("\n=== POLICY SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== APEX-P K-ONLY ACTION DISTRIBUTION ===")
    print(dist.to_string(index=False))

    print("\n=== APEX-P K-ONLY SAFE ACTION DISTRIBUTION ===")
    print(safe_dist.to_string(index=False))

    print("\nWrote:", out_dir)


if __name__ == "__main__":
    main()
