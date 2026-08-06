#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def pick_actual_col(df):
    for c in ["actual_tps", "target_tokens_per_sec", "tokens_per_sec"]:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find actual TPS column. Columns:\n{list(df.columns)}")


def pick_prompt_col(df):
    for c in ["prompt_hash", "prompt_id", "req_id"]:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find prompt id column. Columns:\n{list(df.columns)}")


def pick_score_col(df, user_col=None):
    if user_col:
        if user_col not in df.columns:
            raise SystemExit(f"--score-col {user_col} not in columns:\n{list(df.columns)}")
        return user_col

    preferred = [
        "full_score",
        "apexp_full_score",
        "score_full",
        "utility_full",
        "pred_utility",
        "predicted_utility",
        "policy_score",
        "score",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(x in lc for x in ["score", "utility"]):
            if not any(x in lc for x in ["actual", "target", "oracle"]):
                if pd.api.types.is_numeric_dtype(df[c]):
                    candidates.append(c)

    if candidates:
        print("Auto-detected possible score columns:", candidates)
        return candidates[0]

    # Last-resort: build utility if model emitted expected accepted length and cost.
    exp_cols = [c for c in df.columns if "expected" in c.lower() and "accepted" in c.lower()]
    cost_cols = [c for c in df.columns if "cost" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    if exp_cols and cost_cols:
        e = exp_cols[0]
        cost = cost_cols[0]
        df["derived_fast_utility"] = df[e] / np.exp(df[cost].clip(-20, 20))
        print(f"Derived score from {e} / exp({cost})")
        return "derived_fast_utility"

    print("\nCould not auto-detect score column.")
    print("\nNumeric columns:")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            print("  ", c)
    raise SystemExit("Pass --score-col explicitly using one of the columns above.")


def one_per_prompt_max(df, prompt_col, score_col):
    idx = df.groupby(prompt_col)[score_col].idxmax()
    return df.loc[idx].copy()


def summarize_policy(name, selected, actual_col, ar_tps, oracle_tps):
    mean_tps = float(selected[actual_col].mean())
    return {
        "policy": name,
        "n_requests": int(selected.shape[0]),
        "mean_tps": mean_tps,
        "speedup_vs_ar": mean_tps / ar_tps if ar_tps else np.nan,
        "oracle_fraction": mean_tps / oracle_tps if oracle_tps else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-table", required=True)
    ap.add_argument("--slow-router-dir", required=True)
    ap.add_argument("--train-global-action-stats", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--prior-floor", type=float, default=0.95)
    ap.add_argument("--prior-beta", type=float, default=1.0)
    args = ap.parse_args()

    cand = pd.read_csv(args.candidate_table)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    actual_col = pick_actual_col(cand)
    prompt_col = pick_prompt_col(cand)
    score_col = pick_score_col(cand, args.score_col)

    required = [prompt_col, "method", "k_requested", actual_col, score_col]
    missing = [c for c in required if c not in cand.columns]
    if missing:
        raise SystemExit(f"Missing required columns in candidate table: {missing}")

    print("Using:")
    print("  prompt_col:", prompt_col)
    print("  actual_col:", actual_col)
    print("  score_col :", score_col)

    # Slow router selected actions
    slow_dir = Path(args.slow_router_dir)
    slow_scored = pd.read_csv(slow_dir / "test_candidates_scored.csv")
    slow_actual_col = pick_actual_col(slow_scored)
    slow_prompt_col = pick_prompt_col(slow_scored)

    if "pred_tps" not in slow_scored.columns:
        pred_cols = [c for c in slow_scored.columns if "pred" in c.lower()]
        raise SystemExit(f"Slow router file has no pred_tps. Prediction columns: {pred_cols}")

    slow_sel = one_per_prompt_max(slow_scored, slow_prompt_col, "pred_tps")
    slow_sel = slow_sel[[slow_prompt_col, "method", "k_requested", slow_actual_col]].rename(
        columns={
            slow_prompt_col: prompt_col,
            "method": "slow_method",
            "k_requested": "slow_k",
            slow_actual_col: "slow_actual_tps",
        }
    )

    # Training global action priors
    prior = pd.read_csv(args.train_global_action_stats)
    prior_actual_col = pick_actual_col(prior) if actual_col not in prior.columns else actual_col
    if prior_actual_col != "actual_tps" and "actual_tps" not in prior.columns:
        prior = prior.rename(columns={prior_actual_col: "train_prior_tps"})
    elif "actual_tps" in prior.columns:
        prior = prior.rename(columns={"actual_tps": "train_prior_tps"})

    prior = prior[["method", "k_requested", "train_prior_tps"]].copy()

    # Merge slow decision and priors
    cand = cand.merge(slow_sel, on=prompt_col, how="inner")
    cand = cand.merge(prior, on=["method", "k_requested"], how="left")

    slow_prior = prior.rename(
        columns={
            "method": "slow_method",
            "k_requested": "slow_k",
            "train_prior_tps": "slow_train_prior_tps",
        }
    )
    cand = cand.merge(slow_prior, on=["slow_method", "slow_k"], how="left")

    # Baselines from candidate table
    oracle = one_per_prompt_max(cand, prompt_col, actual_col)

    fixed_ngram16 = cand[(cand["method"] == "ngram_sd") & (cand["k_requested"] == 16)].copy()
    fixed_ngram16 = one_per_prompt_max(fixed_ngram16, prompt_col, actual_col)

    fixed_eagle4 = cand[(cand["method"] == "eagle3") & (cand["k_requested"] == 4)].copy()
    fixed_eagle4 = one_per_prompt_max(fixed_eagle4, prompt_col, actual_col)

    # AR TPS: infer from existing fast replay policy summary if available, else from oracle speedup unavailable.
    # We infer AR from slow router summary if available.
    slow_policy = pd.read_csv(slow_dir / "policy_summary.csv")
    ar_tps = None
    if "speedup_vs_ar" in slow_policy.columns:
        row = slow_policy[slow_policy["policy"] == "request_prompt_router_no_workload"]
        if not row.empty:
            ar_tps = float(row.iloc[0]["mean_tps"]) / float(row.iloc[0]["speedup_vs_ar"])

    if ar_tps is None:
        ar_tps = float(fixed_ngram16[actual_col].mean()) / 2.9260545605002526
        print("WARNING: using fallback AR TPS estimate:", ar_tps)

    oracle_tps = float(oracle[actual_col].mean())

    # Slow router selected actual rows from candidate table, to ensure same candidate source
    slow_rows = cand[
        (cand["method"] == cand["slow_method"])
        & (cand["k_requested"].astype(int) == cand["slow_k"].astype(int))
    ].copy()
    slow_rows = one_per_prompt_max(slow_rows, prompt_col, actual_col)

    # Bad current behavior: fast chooses from all methods/k
    fast_any = one_per_prompt_max(cand, prompt_col, score_col)

    # Correct design: fast chooses k only inside slow-selected method
    within_method = cand[cand["method"] == cand["slow_method"]].copy()
    fast_k_within_slow = one_per_prompt_max(within_method, prompt_col, score_col)

    # Safe design: fast chooses k inside slow method but only among k whose train prior
    # is close to the slow action's train prior. This prevents obvious bad high-k choices.
    safe = within_method.copy()
    safe["allowed_by_prior"] = (
        safe["train_prior_tps"].fillna(-1)
        >= args.prior_floor * safe["slow_train_prior_tps"].fillna(1e9)
    )

    # Always allow the original slow action as fallback.
    is_slow_action = safe["k_requested"].astype(int) == safe["slow_k"].astype(int)
    safe["allowed_by_prior"] = safe["allowed_by_prior"] | is_slow_action

    safe_allowed = safe[safe["allowed_by_prior"]].copy()

    # Rank-normalized combined score:
    # - fast score gives per-prompt signal
    # - train prior prevents pathological choices
    safe_allowed["fast_rank"] = safe_allowed.groupby(prompt_col)[score_col].rank(pct=True)
    safe_allowed["prior_rank"] = safe_allowed["train_prior_tps"].rank(pct=True)
    safe_allowed["safe_score"] = safe_allowed["fast_rank"] + args.prior_beta * safe_allowed["prior_rank"]

    fast_safe = one_per_prompt_max(safe_allowed, prompt_col, "safe_score")

    rows = []
    rows.append(summarize_policy("oracle_replay", oracle, actual_col, ar_tps, oracle_tps))
    rows.append(summarize_policy("slow_router_replay", slow_rows, actual_col, ar_tps, oracle_tps))
    rows.append(summarize_policy("fixed_ngram_k16", fixed_ngram16, actual_col, ar_tps, oracle_tps))
    rows.append(summarize_policy("fixed_eagle3_k4", fixed_eagle4, actual_col, ar_tps, oracle_tps))
    rows.append(summarize_policy("fast_raw_any_method_BAD", fast_any, actual_col, ar_tps, oracle_tps))
    rows.append(summarize_policy("slow_plus_fast_k_within_slow_method", fast_k_within_slow, actual_col, ar_tps, oracle_tps))
    rows.append(summarize_policy(f"slow_plus_fast_safe_prior_floor_{args.prior_floor}", fast_safe, actual_col, ar_tps, oracle_tps))

    summary = pd.DataFrame(rows).sort_values("mean_tps", ascending=False)
    summary.to_csv(out / "safe_slow_fast_policy_summary.csv", index=False)

    for name, df in [
        ("selected_oracle_replay.csv", oracle),
        ("selected_slow_router_replay.csv", slow_rows),
        ("selected_fast_raw_any_method_BAD.csv", fast_any),
        ("selected_slow_plus_fast_k_within_slow_method.csv", fast_k_within_slow),
        ("selected_slow_plus_fast_safe.csv", fast_safe),
    ]:
        df.to_csv(out / name, index=False)

    dist_rows = []
    for name, df in [
        ("slow_router_replay", slow_rows),
        ("fast_raw_any_method_BAD", fast_any),
        ("slow_plus_fast_k_within_slow_method", fast_k_within_slow),
        ("slow_plus_fast_safe", fast_safe),
    ]:
        d = df.groupby(["method", "k_requested"]).size().reset_index(name="count")
        d["policy"] = name
        d["fraction"] = d["count"] / d["count"].sum()
        dist_rows.append(d)

    dist = pd.concat(dist_rows, ignore_index=True)
    dist.to_csv(out / "safe_slow_fast_action_distribution.csv", index=False)

    print("\n=== SAFE SLOW+FAST POLICY SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== SAFE SLOW+FAST ACTION DISTRIBUTION ===")
    print(dist.sort_values(["policy", "count"], ascending=[True, False]).to_string(index=False))

    print("\nWrote:", out)


if __name__ == "__main__":
    main()
