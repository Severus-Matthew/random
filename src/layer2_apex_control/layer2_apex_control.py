#!/usr/bin/env python3
"""
Layer 2: APEX-Control offline policy evaluation.

Input:
  results/apex_layer1/layer1_request_survival.csv

Output:
  policy_decisions.csv
  controller_comparison.csv
  controller_by_workload.csv
  figures/

This evaluates whether accepted-length survival + cost can choose better
speculative configurations than fixed-k, workload-tuned, entropy/repetition,
and BanditSpec-style baselines.

Important:
  This is offline request-level replay. It does not require perfect block-level
  Layer 0. It uses the request-level survival traces available now.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_KEEP = {"eagle3", "ngram_sd", "draft_sd", "ar", "autoregressive"}
NON_AR_METHODS = {"eagle3", "ngram_sd", "draft_sd"}


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"}:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=-1):
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return default
    return int(v)


def normalize_method(m):
    m = str(m).strip()
    if m in {"AR", "autoregressive"}:
        return "ar"
    return m


def add_request_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "request_id" in df.columns:
        rid = df["request_id"].astype(str)
    elif "id" in df.columns:
        rid = df["id"].astype(str)
    elif "request_index" in df.columns:
        rid = df["request_index"].astype(str)
    else:
        rid = df.groupby(["workload"]).cumcount().astype(str)

    df["request_key"] = df["workload"].astype(str) + "::" + rid
    return df


def add_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer speedup_vs_ar if already present.
    Otherwise compute speedup = tokens_per_sec / median AR tokens_per_sec by workload.
    """
    df = df.copy()

    if "speedup_vs_ar" in df.columns:
        df["realized_speedup"] = pd.to_numeric(df["speedup_vs_ar"], errors="coerce")
    else:
        df["realized_speedup"] = np.nan

    if df["realized_speedup"].isna().all():
        if "tokens_per_sec" not in df.columns:
            raise ValueError("Need either speedup_vs_ar or tokens_per_sec to compute speedup.")

        ar = df[df["method"].isin(["ar", "autoregressive"])].copy()
        if len(ar) == 0:
            raise ValueError("No AR rows found and speedup_vs_ar missing.")

        ar_base = (
            ar.groupby("workload")["tokens_per_sec"]
            .median()
            .rename("ar_tokens_per_sec")
            .reset_index()
        )
        df = df.merge(ar_base, on="workload", how="left")
        df["realized_speedup"] = df["tokens_per_sec"] / df["ar_tokens_per_sec"]

    return df


def choose_expected_length_col(df: pd.DataFrame) -> str:
    """
    Choose the best available request-level accepted-length estimate.
    """
    candidates = [
        "apex_expected_len_counter",
        "accepted_per_draft",
        "oracle_expected_tokens",
        "apex_expected_len_survival_observed",
        "observed_survival_E_L",
        "expected_len_observed",
    ]
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
            return c
    raise ValueError("No accepted-length column found.")


def preprocess(input_csv: Path, temperature_filter: str = "0") -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(input_csv)

    required = ["workload", "method"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["method"] = df["method"].map(normalize_method)

    if "k" not in df.columns:
        # Some older logs may call it depth.
        if "depth" in df.columns:
            df["k"] = df["depth"]
        else:
            raise ValueError("Missing k/depth column.")

    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")

    # Filter methods.
    df = df[df["method"].isin(METHOD_KEEP)].copy()

    # Optional temperature filter.
    if temperature_filter != "all" and "temperature" in df.columns:
        t = float(temperature_filter)
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df = df[np.isclose(df["temperature"].fillna(t), t)].copy()

    # Numeric columns.
    for c in [
        "tokens_per_sec",
        "latency_s",
        "mean_entropy",
        "repetition_density",
        "accepted_per_draft",
        "acceptance_rate",
        "apex_expected_len_counter",
        "apex_expected_len_survival_observed",
        "oracle_expected_tokens",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_request_key(df)
    df = add_speedup(df)

    expected_col = choose_expected_length_col(df)
    df["realized_E_L"] = pd.to_numeric(df[expected_col], errors="coerce")

    # If acceptance_rate missing, use accepted_per_draft/k as approximate rate.
    if "acceptance_rate" not in df.columns or df["acceptance_rate"].isna().all():
        df["acceptance_rate"] = df["realized_E_L"] / df["k"].astype(float)
    df["acceptance_rate"] = pd.to_numeric(df["acceptance_rate"], errors="coerce")

    # A simple cost proxy. Lower is better.
    if "latency_s" in df.columns:
        df["cost_s"] = pd.to_numeric(df["latency_s"], errors="coerce")
    else:
        df["cost_s"] = np.nan

    # If no latency, use inverse throughput proxy.
    if df["cost_s"].isna().all():
        df["cost_s"] = 1.0 / pd.to_numeric(df["tokens_per_sec"], errors="coerce")

    # Candidate rows.
    df["is_candidate"] = df["method"].isin(NON_AR_METHODS) & df["k"].notna()

    return df, expected_col


def make_state_buckets(train_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add entropy/repetition buckets based on train quantiles.
    """
    df = full_df.copy()

    if "mean_entropy" not in df.columns:
        df["mean_entropy"] = np.nan
    if "repetition_density" not in df.columns:
        df["repetition_density"] = np.nan

    def make_bucket(col, labels):
        train_vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if len(train_vals) < 10 or train_vals.nunique() < 3:
            return pd.Series(["unknown"] * len(df), index=df.index)

        q1, q2 = train_vals.quantile([0.33, 0.66]).values

        def bucket(v):
            if not np.isfinite(v):
                return "unknown"
            if v <= q1:
                return labels[0]
            if v <= q2:
                return labels[1]
            return labels[2]

        return pd.to_numeric(df[col], errors="coerce").map(bucket)

    df["entropy_bucket"] = make_bucket("mean_entropy", ["low_entropy", "mid_entropy", "high_entropy"])
    df["repetition_bucket"] = make_bucket("repetition_density", ["low_rep", "mid_rep", "high_rep"])
    df["state_bucket"] = df["entropy_bucket"].astype(str) + "|" + df["repetition_bucket"].astype(str)

    return df


def train_test_split_by_request(df: pd.DataFrame, seed: int, train_frac: float = 0.7):
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(df["request_key"].dropna().unique()))
    rng.shuffle(keys)
    n_train = max(1, int(len(keys) * train_frac))
    train_keys = set(keys[:n_train])
    test_keys = set(keys[n_train:])
    if len(test_keys) == 0:
        test_keys = set(keys[-max(1, len(keys)//5):])
        train_keys = set(keys) - test_keys
    return train_keys, test_keys


def median_or_nan(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.median())


def build_train_tables(train: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cand = train[train["is_candidate"]].copy()

    # Main arm statistics.
    arm_cols = ["workload", "method", "k"]
    arm_stats = (
        cand.groupby(arm_cols)
        .agg(
            n=("realized_speedup", "count"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
            median_E_L=("realized_E_L", "median"),
            median_cost_s=("cost_s", "median"),
            median_latency_s=("latency_s", "median") if "latency_s" in cand.columns else ("cost_s", "median"),
            p95_latency_s=("latency_s", lambda x: np.nanpercentile(pd.to_numeric(x, errors="coerce").dropna(), 95) if pd.to_numeric(x, errors="coerce").notna().sum() else np.nan) if "latency_s" in cand.columns else ("cost_s", "median"),
        )
        .reset_index()
    )
    arm_stats["utility"] = arm_stats["median_E_L"] / arm_stats["median_cost_s"].replace(0, np.nan)

    # Global fallback by method/k.
    global_arm = (
        cand.groupby(["method", "k"])
        .agg(
            n=("realized_speedup", "count"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
            median_E_L=("realized_E_L", "median"),
            median_cost_s=("cost_s", "median"),
        )
        .reset_index()
    )
    global_arm["utility"] = global_arm["median_E_L"] / global_arm["median_cost_s"].replace(0, np.nan)

    # State-bucket posterior/cost table.
    bucket_cols = ["workload", "state_bucket", "method", "k"]
    bucket_stats = (
        cand.groupby(bucket_cols)
        .agg(
            n=("realized_speedup", "count"),
            median_speedup=("realized_speedup", "median"),
            median_E_L=("realized_E_L", "median"),
            median_cost_s=("cost_s", "median"),
            p95_cost_s=("cost_s", lambda x: np.nanpercentile(pd.to_numeric(x, errors="coerce").dropna(), 95) if pd.to_numeric(x, errors="coerce").notna().sum() else np.nan),
        )
        .reset_index()
    )
    bucket_stats["utility"] = bucket_stats["median_E_L"] / bucket_stats["median_cost_s"].replace(0, np.nan)

    # State fallback across workloads.
    bucket_global = (
        cand.groupby(["state_bucket", "method", "k"])
        .agg(
            n=("realized_speedup", "count"),
            median_speedup=("realized_speedup", "median"),
            median_E_L=("realized_E_L", "median"),
            median_cost_s=("cost_s", "median"),
            p95_cost_s=("cost_s", lambda x: np.nanpercentile(pd.to_numeric(x, errors="coerce").dropna(), 95) if pd.to_numeric(x, errors="coerce").notna().sum() else np.nan),
        )
        .reset_index()
    )
    bucket_global["utility"] = bucket_global["median_E_L"] / bucket_global["median_cost_s"].replace(0, np.nan)

    return {
        "arm_stats": arm_stats,
        "global_arm": global_arm,
        "bucket_stats": bucket_stats,
        "bucket_global": bucket_global,
    }


def _lookup_stat(table: pd.DataFrame, filters: Dict, value: str) -> float:
    sub = table
    for k, v in filters.items():
        if k not in sub.columns:
            return np.nan
        sub = sub[sub[k] == v]
    if len(sub) == 0 or value not in sub.columns:
        return np.nan
    vals = pd.to_numeric(sub[value], errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float(vals.iloc[0])


def arm_pred_utility(row, tables, lambda_tail=0.0):
    """
    APEX survival-cost score:
      utility = predicted E[L] / predicted cost - lambda_tail * predicted tail cost
    Uses held-out train statistics, with fallbacks.
    """
    workload = row["workload"]
    method = row["method"]
    k = int(row["k"])
    state_bucket = row.get("state_bucket", "unknown")

    # 1. workload + state bucket + method/k
    filters = {"workload": workload, "state_bucket": state_bucket, "method": method, "k": k}
    E = _lookup_stat(tables["bucket_stats"], filters, "median_E_L")
    C = _lookup_stat(tables["bucket_stats"], filters, "median_cost_s")
    P95 = _lookup_stat(tables["bucket_stats"], filters, "p95_cost_s")

    # 2. state bucket + method/k
    if not np.isfinite(E) or not np.isfinite(C):
        filters = {"state_bucket": state_bucket, "method": method, "k": k}
        E = _lookup_stat(tables["bucket_global"], filters, "median_E_L")
        C = _lookup_stat(tables["bucket_global"], filters, "median_cost_s")
        P95 = _lookup_stat(tables["bucket_global"], filters, "p95_cost_s")

    # 3. workload + method/k
    if not np.isfinite(E) or not np.isfinite(C):
        filters = {"workload": workload, "method": method, "k": k}
        E = _lookup_stat(tables["arm_stats"], filters, "median_E_L")
        C = _lookup_stat(tables["arm_stats"], filters, "median_cost_s")
        P95 = _lookup_stat(tables["arm_stats"], filters, "p95_latency_s")

    # 4. method/k global
    if not np.isfinite(E) or not np.isfinite(C):
        filters = {"method": method, "k": k}
        E = _lookup_stat(tables["global_arm"], filters, "median_E_L")
        C = _lookup_stat(tables["global_arm"], filters, "median_cost_s")
        P95 = np.nan

    if not np.isfinite(E) or not np.isfinite(C) or C <= 0:
        return -np.inf, np.nan, np.nan

    util = E / C
    if np.isfinite(P95):
        util -= lambda_tail * P95

    return float(util), float(E), float(C)


def select_best_row(candidates: pd.DataFrame, score_col: str):
    if len(candidates) == 0:
        return None
    vals = pd.to_numeric(candidates[score_col], errors="coerce")
    if vals.notna().sum() == 0:
        return None
    idx = vals.idxmax()
    return candidates.loc[idx]


def choose_policy_for_request(policy: str, candidates: pd.DataFrame, tables: Dict[str, pd.DataFrame], row_state=None, ucb_alpha=1.0, lambda_tail=0.0):
    """
    Return selected row and predicted metadata.
    """
    if len(candidates) == 0:
        return None, {}

    workload = candidates["workload"].iloc[0]

    def choose_filter(method=None, k=None):
        sub = candidates
        if method is not None:
            sub = sub[sub["method"] == method]
        if k is not None:
            sub = sub[sub["k"].astype(int) == int(k)]
        if len(sub) == 0:
            return None
        # if duplicate rows exist, choose the one with median-ish speedup by taking first after sorting
        return sub.iloc[0]

    if policy == "fixed_eagle_k4":
        selected = choose_filter("eagle3", 4)
        return selected, {"predicted_score": np.nan}

    if policy.startswith("fixed_k_"):
        k = int(policy.replace("fixed_k_", ""))
        # Prefer EAGLE if available for fixed-k global baseline, else any method at k.
        selected = choose_filter("eagle3", k)
        if selected is None:
            selected = choose_filter(None, k)
        return selected, {"predicted_score": np.nan}

    if policy == "best_eagle_k_per_workload":
        sub = tables["arm_stats"]
        sub = sub[(sub["workload"] == workload) & (sub["method"] == "eagle3")]
        if len(sub) == 0:
            return None, {}
        k = int(sub.sort_values("median_speedup", ascending=False).iloc[0]["k"])
        selected = choose_filter("eagle3", k)
        return selected, {"chosen_train_k": k}

    if policy == "best_ngram_k_per_workload":
        sub = tables["arm_stats"]
        sub = sub[(sub["workload"] == workload) & (sub["method"] == "ngram_sd")]
        if len(sub) == 0:
            return None, {}
        k = int(sub.sort_values("median_speedup", ascending=False).iloc[0]["k"])
        selected = choose_filter("ngram_sd", k)
        return selected, {"chosen_train_k": k}

    if policy == "best_fixed_method_k_per_workload":
        sub = tables["arm_stats"]
        sub = sub[sub["workload"] == workload]
        if len(sub) == 0:
            return None, {}
        best = sub.sort_values("median_speedup", ascending=False).iloc[0]
        selected = choose_filter(best["method"], int(best["k"]))
        return selected, {"chosen_train_method": best["method"], "chosen_train_k": int(best["k"])}

    if policy == "bandit_ucb_workload":
        sub = tables["arm_stats"].copy()
        sub = sub[sub["workload"] == workload]
        if len(sub) == 0:
            sub = tables["global_arm"].copy()
        if len(sub) == 0:
            return None, {}

        total_n = max(1, sub["n"].sum())
        sub["ucb_score"] = sub["mean_speedup"] + ucb_alpha * np.sqrt(np.log(total_n + 1) / sub["n"].clip(lower=1))
        best = sub.sort_values("ucb_score", ascending=False).iloc[0]
        selected = choose_filter(best["method"], int(best["k"]))
        return selected, {"predicted_score": float(best["ucb_score"]), "chosen_train_method": best["method"], "chosen_train_k": int(best["k"])}

    if policy == "entropy_repetition_rule":
        # Hand-designed rule:
        # high repetition + not high entropy => ngram deeper
        # high entropy => eagle shallow/medium
        # otherwise => eagle best
        entropy_bucket = str(candidates["entropy_bucket"].iloc[0]) if "entropy_bucket" in candidates else "unknown"
        rep_bucket = str(candidates["repetition_bucket"].iloc[0]) if "repetition_bucket" in candidates else "unknown"

        sub = tables["arm_stats"]
        sub = sub[sub["workload"] == workload].copy()

        if rep_bucket == "high_rep" and entropy_bucket != "high_entropy":
            target_method = "ngram_sd"
            allowed_k = [16, 8, 4, 2, 1]
        elif entropy_bucket == "high_entropy":
            target_method = "eagle3"
            allowed_k = [1, 2, 4]
        else:
            target_method = "eagle3"
            allowed_k = [4, 8, 2, 16, 1]

        # Pick best available k for that method among allowed set according to train speedup.
        pref = sub[(sub["method"] == target_method) & (sub["k"].isin(allowed_k))]
        if len(pref) > 0:
            best = pref.sort_values("median_speedup", ascending=False).iloc[0]
            selected = choose_filter(target_method, int(best["k"]))
            return selected, {
                "rule_entropy_bucket": entropy_bucket,
                "rule_repetition_bucket": rep_bucket,
                "chosen_train_method": target_method,
                "chosen_train_k": int(best["k"]),
            }

        # Fallback to best fixed method/k.
        return choose_policy_for_request("best_fixed_method_k_per_workload", candidates, tables)

    if policy == "apex_survival_cost":
        scored = []
        for idx, r in candidates.iterrows():
            util, pred_E, pred_C = arm_pred_utility(r, tables, lambda_tail=lambda_tail)
            scored.append((idx, util, pred_E, pred_C))

        scored_df = pd.DataFrame(scored, columns=["idx", "predicted_score", "predicted_E_L", "predicted_cost_s"])
        scored_df = scored_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["predicted_score"])
        if len(scored_df) == 0:
            return choose_policy_for_request("best_fixed_method_k_per_workload", candidates, tables)

        best = scored_df.sort_values("predicted_score", ascending=False).iloc[0]
        selected = candidates.loc[best["idx"]]
        return selected, {
            "predicted_score": float(best["predicted_score"]),
            "predicted_E_L": float(best["predicted_E_L"]),
            "predicted_cost_s": float(best["predicted_cost_s"]),
        }

    if policy == "oracle_best_per_request":
        selected = select_best_row(candidates, "realized_speedup")
        return selected, {"predicted_score": np.nan}

    if policy == "oracle_survival_cost_per_request":
        # This uses realized request E[L]/cost, so it is an upper-bound diagnostic, not deployable.
        c = candidates.copy()
        c["realized_survival_cost_score"] = c["realized_E_L"] / c["cost_s"].replace(0, np.nan)
        selected = select_best_row(c, "realized_survival_cost_score")
        return selected, {"predicted_score": np.nan}

    raise ValueError(f"Unknown policy: {policy}")


def evaluate_policies(df: pd.DataFrame, seed: int, train_frac: float, ucb_alpha: float, lambda_tail: float):
    # Split by request key so all candidate configs for the same prompt are held out together.
    train_keys, test_keys = train_test_split_by_request(df, seed=seed, train_frac=train_frac)
    train_raw = df[df["request_key"].isin(train_keys)].copy()
    test_raw = df[df["request_key"].isin(test_keys)].copy()

    # Create state buckets using train quantiles but apply to full data.
    df_b = make_state_buckets(train_raw, df)
    train = df_b[df_b["request_key"].isin(train_keys)].copy()
    test = df_b[df_b["request_key"].isin(test_keys)].copy()

    tables = build_train_tables(train)

    policies = [
        "fixed_eagle_k4",
        "fixed_k_1",
        "fixed_k_2",
        "fixed_k_4",
        "fixed_k_8",
        "fixed_k_16",
        "best_eagle_k_per_workload",
        "best_ngram_k_per_workload",
        "best_fixed_method_k_per_workload",
        "entropy_repetition_rule",
        "bandit_ucb_workload",
        "apex_survival_cost",
        "oracle_survival_cost_per_request",
        "oracle_best_per_request",
    ]

    decisions = []

    cand_test = test[test["is_candidate"]].copy()

    # Group by request key. Each group contains available method/k candidates for same prompt.
    for request_key, group in cand_test.groupby("request_key"):
        group = group.copy()
        workload = group["workload"].iloc[0]

        for policy in policies:
            selected, meta = choose_policy_for_request(
                policy=policy,
                candidates=group,
                tables=tables,
                ucb_alpha=ucb_alpha,
                lambda_tail=lambda_tail,
            )

            if selected is None:
                continue

            out = {
                "seed": seed,
                "policy": policy,
                "request_key": request_key,
                "workload": workload,
                "selected_method": selected["method"],
                "selected_k": int(selected["k"]),
                "realized_speedup": selected["realized_speedup"],
                "realized_E_L": selected["realized_E_L"],
                "realized_cost_s": selected["cost_s"],
                "tokens_per_sec": selected.get("tokens_per_sec", np.nan),
                "latency_s": selected.get("latency_s", np.nan),
                "mean_entropy": selected.get("mean_entropy", np.nan),
                "repetition_density": selected.get("repetition_density", np.nan),
                "entropy_bucket": selected.get("entropy_bucket", "unknown"),
                "repetition_bucket": selected.get("repetition_bucket", "unknown"),
            }
            out.update(meta)
            decisions.append(out)

    return pd.DataFrame(decisions)


def summarize(decisions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(decisions) == 0:
        return pd.DataFrame(), pd.DataFrame()

    def p95(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) == 0:
            return np.nan
        return np.nanpercentile(x, 95)

    overall = (
        decisions.groupby("policy")
        .agg(
            n=("realized_speedup", "count"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
            p10_speedup=("realized_speedup", lambda x: np.nanpercentile(pd.to_numeric(x, errors="coerce").dropna(), 10) if pd.to_numeric(x, errors="coerce").notna().sum() else np.nan),
            p90_speedup=("realized_speedup", lambda x: np.nanpercentile(pd.to_numeric(x, errors="coerce").dropna(), 90) if pd.to_numeric(x, errors="coerce").notna().sum() else np.nan),
            mean_E_L=("realized_E_L", "mean"),
            median_E_L=("realized_E_L", "median"),
            mean_latency_s=("latency_s", "mean"),
            p95_latency_s=("latency_s", p95),
            mean_tokens_per_sec=("tokens_per_sec", "mean"),
        )
        .reset_index()
    )

    by_workload = (
        decisions.groupby(["workload", "policy"])
        .agg(
            n=("realized_speedup", "count"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
            mean_E_L=("realized_E_L", "mean"),
            mean_latency_s=("latency_s", "mean"),
            p95_latency_s=("latency_s", p95),
        )
        .reset_index()
    )

    # Add regret vs oracle by matching each seed/request to oracle best.
    oracle = decisions[decisions["policy"] == "oracle_best_per_request"][
        ["seed", "request_key", "realized_speedup"]
    ].rename(columns={"realized_speedup": "oracle_speedup"})

    d = decisions.merge(oracle, on=["seed", "request_key"], how="left")
    d["regret_vs_oracle"] = d["oracle_speedup"] - d["realized_speedup"]

    regret = (
        d.groupby("policy")
        .agg(
            mean_regret_vs_oracle=("regret_vs_oracle", "mean"),
            median_regret_vs_oracle=("regret_vs_oracle", "median"),
        )
        .reset_index()
    )

    overall = overall.merge(regret, on="policy", how="left")

    return overall, by_workload


def plot_bar(df, x_col, y_col, out_path, title, ylabel, sort=True):
    d = df[[x_col, y_col]].dropna().copy()
    if sort:
        d = d.sort_values(y_col, ascending=True)

    plt.figure(figsize=(10, max(4, 0.35 * len(d))))
    plt.barh(d[x_col].astype(str), d[y_col].astype(float))
    plt.xlabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_k_distribution(decisions, out_path):
    d = decisions[decisions["policy"].isin(["apex_survival_cost", "entropy_repetition_rule", "bandit_ucb_workload"])].copy()
    if len(d) == 0:
        return
    pivot = (
        d.groupby(["policy", "selected_k"])
        .size()
        .reset_index(name="count")
    )
    policies = list(pivot["policy"].drop_duplicates())
    ks = sorted(pivot["selected_k"].dropna().unique())

    x = np.arange(len(ks))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for i, p in enumerate(policies):
        vals = []
        for k in ks:
            sub = pivot[(pivot["policy"] == p) & (pivot["selected_k"] == k)]
            vals.append(int(sub["count"].iloc[0]) if len(sub) else 0)
        plt.bar(x + (i - len(policies)/2) * width, vals, width=width, label=p)

    plt.xticks(x, [str(k) for k in ks])
    plt.xlabel("Selected k")
    plt.ylabel("Number of requests")
    plt.title("Controller-selected depth distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_predicted_vs_realized(decisions, out_path):
    d = decisions[decisions["policy"] == "apex_survival_cost"].copy()
    if len(d) == 0 or "predicted_E_L" not in d.columns:
        return
    d["predicted_E_L"] = pd.to_numeric(d["predicted_E_L"], errors="coerce")
    d["realized_E_L"] = pd.to_numeric(d["realized_E_L"], errors="coerce")
    d = d.dropna(subset=["predicted_E_L", "realized_E_L"])
    if len(d) == 0:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(d["predicted_E_L"], d["realized_E_L"], s=10, alpha=0.5)
    lo = min(d["predicted_E_L"].min(), d["realized_E_L"].min())
    hi = max(d["predicted_E_L"].max(), d["realized_E_L"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Predicted E[L]")
    plt.ylabel("Realized E[L]")
    plt.title("APEX predicted vs realized accepted length")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_workload_heatmap(by_workload, out_path):
    d = by_workload.copy()
    keep = [
        "fixed_eagle_k4",
        "best_eagle_k_per_workload",
        "best_ngram_k_per_workload",
        "best_fixed_method_k_per_workload",
        "entropy_repetition_rule",
        "bandit_ucb_workload",
        "apex_survival_cost",
        "oracle_best_per_request",
    ]
    d = d[d["policy"].isin(keep)]

    pivot = d.pivot_table(index="workload", columns="policy", values="mean_speedup", aggfunc="mean")
    if pivot.empty:
        return

    plt.figure(figsize=(12, max(4, 0.45 * len(pivot))))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Mean speedup vs AR")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.title("Layer 2 policy speedup by workload")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_figures(decisions, overall, by_workload, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    if len(overall):
        plot_bar(
            overall,
            "policy",
            "mean_speedup",
            fig_dir / "layer2_policy_mean_speedup.png",
            "Layer 2 policy comparison",
            "Mean speedup vs AR",
        )
        plot_bar(
            overall,
            "policy",
            "mean_regret_vs_oracle",
            fig_dir / "layer2_policy_regret_vs_oracle.png",
            "Regret relative to per-request oracle",
            "Mean oracle regret",
            sort=False,
        )

    plot_k_distribution(decisions, fig_dir / "layer2_selected_k_distribution.png")
    plot_predicted_vs_realized(decisions, fig_dir / "layer2_apex_predicted_vs_realized_E_L.png")
    plot_workload_heatmap(by_workload, fig_dir / "layer2_speedup_by_workload_policy.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to layer1_request_survival.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--temperature", default="0", help="'0' or 'all'")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--ucb-alpha", type=float, default=1.0)
    ap.add_argument("--lambda-tail", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, expected_col = preprocess(Path(args.input), temperature_filter=args.temperature)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    all_decisions = []

    for seed in seeds:
        dec = evaluate_policies(
            df,
            seed=seed,
            train_frac=args.train_frac,
            ucb_alpha=args.ucb_alpha,
            lambda_tail=args.lambda_tail,
        )
        all_decisions.append(dec)

    decisions = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    overall, by_workload = summarize(decisions)

    decisions.to_csv(out_dir / "policy_decisions.csv", index=False)
    overall.to_csv(out_dir / "controller_comparison.csv", index=False)
    by_workload.to_csv(out_dir / "controller_by_workload.csv", index=False)

    diagnostics = {
        "input": str(args.input),
        "rows_total": int(len(df)),
        "candidate_rows": int(df["is_candidate"].sum()),
        "expected_length_column": expected_col,
        "temperature_filter": args.temperature,
        "seeds": seeds,
        "train_frac": args.train_frac,
        "ucb_alpha": args.ucb_alpha,
        "lambda_tail": args.lambda_tail,
        "policies": sorted(decisions["policy"].unique().tolist()) if len(decisions) else [],
        "requests_evaluated": int(decisions["request_key"].nunique()) if len(decisions) else 0,
    }
    with open(out_dir / "layer2_diagnostics.json", "w") as f:
        import json
        json.dump(diagnostics, f, indent=2)

    make_figures(decisions, overall, by_workload, out_dir / "figures")

    print("Layer 2 complete")
    print(f"Input: {args.input}")
    print(f"Output: {out_dir}")
    print(f"Expected length column: {expected_col}")
    print(f"Rows: {len(df)} candidate rows: {int(df['is_candidate'].sum())}")
    print(f"Decisions: {len(decisions)}")
    print(f"Wrote: {out_dir / 'controller_comparison.csv'}")
    print(f"Wrote: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
