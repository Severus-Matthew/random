#!/usr/bin/env python3
"""
Layer 2 stress tests + 2C-lite learned hazard model.

This script uses request-level Layer 1 survival data, not strict block-level traces.

Input:
  results/apex_layer1/layer1_request_survival.csv

Outputs:
  results/apex_layer2_stress_2c/
    decisions_all.csv
    controller_comparison_full.csv
    controller_comparison_common_support.csv
    paired_bootstrap_ci.csv
    per_workload_main_policies.csv
    selected_method_k_distribution.csv
    ablation_summary.csv
    hazard_2c_lite_predictions.csv
    hazard_2c_lite_calibration.csv
    figures/

What this script tests:
  - whether APEX survives paired bootstrap CI checks
  - whether APEX wins per workload
  - whether APEX is actually adapting method/k
  - whether entropy matters
  - whether repetition matters
  - whether survival/E[L] matters
  - whether learned request-level hazard prediction improves over bucketed APEX

Important:
  This is 2C-lite, not true block-level censored 2C.
  True 2C requires per-block first rejection traces from vLLM.
"""

import argparse
import json
import math
import os
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MAIN_POLICIES = [
    "best_fixed_method_k_per_workload",
    "entropy_repetition_rule",
    "bandit_ucb_workload",
    "apex_bucketed_full",
    "apex_no_entropy",
    "apex_no_repetition",
    "apex_no_survival",
    "apex_2c_lite_hazard",
    "oracle_best_per_request",
]


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_method(x):
    x = str(x).strip()
    if x.lower() in {"ar", "autoregressive"}:
        return "ar"
    return x


def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def save_2c_lite_models(models_by_seed, out_dir: Path):
    """
    Save trained 2C-lite hazard models.

    models_by_seed is a list of:
      {
        "seed": seed,
        "models": {position: sklearn_model},
        "meta": metadata_dict
      }
    """
    model_dir = out_dir / "models_2c_lite"
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for item in models_by_seed:
        seed = item["seed"]
        models = item["models"]
        meta = item["meta"]

        seed_dir = model_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for pos, model in models.items():
            path = seed_dir / f"hazard_pos_{pos}.joblib"
            joblib.dump(model, path)
            manifest.append({
                "seed": seed,
                "position": int(pos),
                "model_path": str(path),
                "target": f"hazard_pos_{pos}",
                "model_type": "sklearn HistGradientBoostingRegressor"
            })

        with open(seed_dir / "feature_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    with open(model_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

def add_request_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rid_col = pick_first_existing(
        df,
        [
            "request_id",
            "sample_id",
            "prompt_id",
            "example_id",
            "id",
            "request_index",
            "idx",
        ],
    )

    if rid_col is not None:
        rid = df[rid_col].astype(str)
    else:
        # Last fallback. This is weaker because exact cross-method prompt alignment
        # may not be guaranteed, but it keeps the script runnable.
        group_cols = ["workload", "method", "k"]
        df["_local_idx_fallback"] = df.groupby(group_cols).cumcount()
        rid = df["_local_idx_fallback"].astype(str)

    df["request_key"] = df["workload"].astype(str) + "::" + rid
    return df


def add_speedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "speedup_vs_ar" in df.columns:
        df["realized_speedup"] = safe_num(df["speedup_vs_ar"])
        if df["realized_speedup"].notna().sum() > 0:
            return df

    if "tokens_per_sec" not in df.columns:
        raise ValueError("Need either speedup_vs_ar or tokens_per_sec.")

    ar = df[df["method"] == "ar"].copy()
    if len(ar) == 0:
        raise ValueError("No AR rows found and speedup_vs_ar missing.")

    ar_base = (
        ar.groupby("workload")["tokens_per_sec"]
        .median()
        .rename("ar_tokens_per_sec")
        .reset_index()
    )
    df = df.merge(ar_base, on="workload", how="left")
    df["realized_speedup"] = safe_num(df["tokens_per_sec"]) / safe_num(df["ar_tokens_per_sec"])
    return df


def choose_el_col(df: pd.DataFrame) -> str:
    candidates = [
        "apex_expected_len_counter",
        "apex_expected_len_survival_observed",
        "accepted_per_draft",
        "oracle_expected_tokens",
        "expected_len_observed",
        "observed_survival_E_L",
    ]
    for c in candidates:
        if c in df.columns and safe_num(df[c]).notna().sum() > 0:
            return c
    raise ValueError("Could not find accepted-length / E[L] column.")


def preprocess(path: Path, temperature: str) -> Tuple[pd.DataFrame, str, List[int]]:
    df = pd.read_csv(path)

    for c in ["workload", "method"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}")

    if "k" not in df.columns:
        if "depth" in df.columns:
            df["k"] = df["depth"]
        else:
            raise ValueError("Missing k/depth column.")

    df["method"] = df["method"].map(normalize_method)
    df["k"] = safe_num(df["k"]).astype("Int64")

    if temperature != "all" and "temperature" in df.columns:
        df["temperature"] = safe_num(df["temperature"])
        target = float(temperature)
        df = df[np.isclose(df["temperature"].fillna(target), target)].copy()

    numeric_cols = [
        "tokens_per_sec",
        "latency_s",
        "mean_entropy",
        "repetition_density",
        "acceptance_rate",
        "accepted_per_draft",
        "apex_expected_len_counter",
        "apex_expected_len_survival_observed",
        "oracle_expected_tokens",
        "temperature",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = safe_num(df[c])

    df = add_request_key(df)
    df = add_speedup(df)

    el_col = choose_el_col(df)
    df["realized_E_L"] = safe_num(df[el_col])

    if "latency_s" in df.columns:
        df["cost_s"] = safe_num(df["latency_s"])
    else:
        df["cost_s"] = np.nan

    if df["cost_s"].isna().all():
        if "tokens_per_sec" not in df.columns:
            raise ValueError("Need latency_s or tokens_per_sec for cost.")
        df["cost_s"] = 1.0 / safe_num(df["tokens_per_sec"])

    if "mean_entropy" not in df.columns:
        df["mean_entropy"] = np.nan
    if "repetition_density" not in df.columns:
        df["repetition_density"] = np.nan

    df["is_candidate"] = df["method"].isin(["eagle3", "ngram_sd", "draft_sd"]) & df["k"].notna()

    hazard_positions = []
    for c in df.columns:
        if c.startswith("hazard_pos_"):
            try:
                hazard_positions.append(int(c.replace("hazard_pos_", "")))
            except Exception:
                pass
    hazard_positions = sorted(hazard_positions)

    if not hazard_positions:
        # Try deriving hazards from survival columns.
        survival_positions = []
        for c in df.columns:
            if c.startswith("survival_pos_"):
                try:
                    survival_positions.append(int(c.replace("survival_pos_", "")))
                except Exception:
                    pass
        survival_positions = sorted(survival_positions)
        for j in survival_positions:
            sj = safe_num(df[f"survival_pos_{j}"])
            if j == 0:
                pj = sj
            else:
                prev = safe_num(df[f"survival_pos_{j-1}"])
                pj = sj / prev.replace(0, np.nan)
            hj = 1.0 - pj
            df[f"hazard_pos_{j}"] = hj.clip(0.0, 1.0)
        hazard_positions = survival_positions

    if not hazard_positions:
        raise ValueError("No hazard_pos_j or survival_pos_j columns found. Cannot do 2C-lite.")

    return df, el_col, hazard_positions


def train_test_split_by_request(df, seed, train_frac):
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(df["request_key"].dropna().unique()))
    rng.shuffle(keys)
    n_train = max(1, int(train_frac * len(keys)))
    train_keys = set(keys[:n_train])
    test_keys = set(keys[n_train:])
    if len(test_keys) == 0:
        test_keys = set(keys[-max(1, len(keys)//5):])
        train_keys = set(keys) - test_keys
    return train_keys, test_keys


def make_bucket_series(train_vals, vals, labels):
    train_vals = safe_num(train_vals).dropna()
    if len(train_vals) < 20 or train_vals.nunique() < 3:
        return pd.Series(["unknown"] * len(vals), index=vals.index)

    q1, q2 = train_vals.quantile([0.33, 0.66]).values

    def b(v):
        if not np.isfinite(v):
            return "unknown"
        if v <= q1:
            return labels[0]
        if v <= q2:
            return labels[1]
        return labels[2]

    return safe_num(vals).map(b)


def add_buckets(full, train):
    full = full.copy()
    full["entropy_bucket"] = make_bucket_series(
        train["mean_entropy"],
        full["mean_entropy"],
        ["low_entropy", "mid_entropy", "high_entropy"],
    )
    full["repetition_bucket"] = make_bucket_series(
        train["repetition_density"],
        full["repetition_density"],
        ["low_rep", "mid_rep", "high_rep"],
    )
    full["state_full"] = full["entropy_bucket"].astype(str) + "|" + full["repetition_bucket"].astype(str)
    full["state_no_entropy"] = full["repetition_bucket"].astype(str)
    full["state_no_repetition"] = full["entropy_bucket"].astype(str)
    full["state_none"] = "all"
    return full


def group_stats(df, state_col):
    cand = df[df["is_candidate"]].copy()

    tables = {}

    tables["state_workload_arm"] = (
        cand.groupby(["workload", state_col, "method", "k"])
        .agg(
            n=("realized_speedup", "count"),
            median_E_L=("realized_E_L", "median"),
            median_cost_s=("cost_s", "median"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
        )
        .reset_index()
        .rename(columns={state_col: "state"})
    )

    tables["state_arm"] = (
        cand.groupby([state_col, "method", "k"])
        .agg(
            n=("realized_speedup", "count"),
            median_E_L=("realized_E_L", "median"),
            median_cost_s=("cost_s", "median"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
        )
        .reset_index()
        .rename(columns={state_col: "state"})
    )

    tables["workload_arm"] = (
        cand.groupby(["workload", "method", "k"])
        .agg(
            n=("realized_speedup", "count"),
            median_E_L=("realized_E_L", "median"),
            median_cost_s=("cost_s", "median"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
        )
        .reset_index()
    )

    tables["arm"] = (
        cand.groupby(["method", "k"])
        .agg(
            n=("realized_speedup", "count"),
            median_E_L=("realized_E_L", "median"),
            median_cost_s=("cost_s", "median"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
        )
        .reset_index()
    )

    return tables


def first_value(table, filters, value):
    sub = table
    for k, v in filters.items():
        if k not in sub.columns:
            return np.nan
        sub = sub[sub[k] == v]
    if len(sub) == 0 or value not in sub.columns:
        return np.nan
    s = safe_num(sub[value]).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.iloc[0])


def predict_bucket_values(row, tables, state_value):
    workload = row["workload"]
    method = row["method"]
    k = int(row["k"])

    # Most specific
    filters = {"workload": workload, "state": state_value, "method": method, "k": k}
    E = first_value(tables["state_workload_arm"], filters, "median_E_L")
    C = first_value(tables["state_workload_arm"], filters, "median_cost_s")
    Sp = first_value(tables["state_workload_arm"], filters, "median_speedup")

    # State global
    if not np.isfinite(E) or not np.isfinite(C):
        filters = {"state": state_value, "method": method, "k": k}
        E = first_value(tables["state_arm"], filters, "median_E_L")
        C = first_value(tables["state_arm"], filters, "median_cost_s")
        Sp = first_value(tables["state_arm"], filters, "median_speedup")

    # Workload arm
    if not np.isfinite(E) or not np.isfinite(C):
        filters = {"workload": workload, "method": method, "k": k}
        E = first_value(tables["workload_arm"], filters, "median_E_L")
        C = first_value(tables["workload_arm"], filters, "median_cost_s")
        Sp = first_value(tables["workload_arm"], filters, "median_speedup")

    # Global arm
    if not np.isfinite(E) or not np.isfinite(C):
        filters = {"method": method, "k": k}
        E = first_value(tables["arm"], filters, "median_E_L")
        C = first_value(tables["arm"], filters, "median_cost_s")
        Sp = first_value(tables["arm"], filters, "median_speedup")

    return E, C, Sp


def choose_max(candidates, score_col):
    if len(candidates) == 0:
        return None
    vals = safe_num(candidates[score_col])
    if vals.notna().sum() == 0:
        return None
    return candidates.loc[vals.idxmax()]


def select_bucket_apex(candidates, tables, state_col, no_survival=False):
    rows = []
    for idx, r in candidates.iterrows():
        E, C, Sp = predict_bucket_values(r, tables, r[state_col])
        if not np.isfinite(C) or C <= 0:
            score = -np.inf
        elif no_survival:
            # Remove survival/E[L]. This tests whether cost-only can explain the gain.
            score = 1.0 / C
        else:
            score = E / C if np.isfinite(E) else -np.inf
        rows.append((idx, score, E, C, Sp))

    s = pd.DataFrame(rows, columns=["idx", "score", "pred_E_L", "pred_cost_s", "pred_speedup"])
    s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    if len(s) == 0:
        return None, {}
    best = s.sort_values("score", ascending=False).iloc[0]
    return candidates.loc[best["idx"]], {
        "predicted_score": float(best["score"]),
        "predicted_E_L": float(best["pred_E_L"]) if np.isfinite(best["pred_E_L"]) else np.nan,
        "predicted_cost_s": float(best["pred_cost_s"]) if np.isfinite(best["pred_cost_s"]) else np.nan,
    }


def select_entropy_rule(candidates, tables):
    workload = candidates["workload"].iloc[0]
    eb = str(candidates["entropy_bucket"].iloc[0])
    rb = str(candidates["repetition_bucket"].iloc[0])

    arm = tables["workload_arm"].copy()
    arm = arm[arm["workload"] == workload]

    if rb == "high_rep" and eb != "high_entropy":
        target_method = "ngram_sd"
        allowed_k = [16, 8, 4, 2, 1]
    elif eb == "high_entropy":
        target_method = "eagle3"
        allowed_k = [1, 2, 4]
    else:
        target_method = "eagle3"
        allowed_k = [4, 8, 2, 16, 1]

    pref = arm[(arm["method"] == target_method) & (arm["k"].isin(allowed_k))]
    if len(pref):
        best = pref.sort_values("median_speedup", ascending=False).iloc[0]
        sub = candidates[
            (candidates["method"] == best["method"]) &
            (candidates["k"].astype(int) == int(best["k"]))
        ]
        if len(sub):
            return sub.iloc[0], {"rule_entropy_bucket": eb, "rule_repetition_bucket": rb}

    return select_best_fixed(candidates, tables)


def select_best_fixed(candidates, tables):
    workload = candidates["workload"].iloc[0]
    arm = tables["workload_arm"]
    sub = arm[arm["workload"] == workload]
    if len(sub) == 0:
        sub = tables["arm"].copy()
    if len(sub) == 0:
        return None, {}
    best = sub.sort_values("median_speedup", ascending=False).iloc[0]
    c = candidates[
        (candidates["method"] == best["method"]) &
        (candidates["k"].astype(int) == int(best["k"]))
    ]
    if len(c) == 0:
        return None, {}
    return c.iloc[0], {"chosen_train_method": best["method"], "chosen_train_k": int(best["k"])}


def select_ucb(candidates, tables, alpha=1.0):
    workload = candidates["workload"].iloc[0]
    arm = tables["workload_arm"].copy()
    arm = arm[arm["workload"] == workload]
    if len(arm) == 0:
        arm = tables["arm"].copy()
    if len(arm) == 0:
        return None, {}

    total_n = max(1, int(arm["n"].sum()))
    arm["ucb_score"] = arm["mean_speedup"] + alpha * np.sqrt(np.log(total_n + 1.0) / arm["n"].clip(lower=1))
    best = arm.sort_values("ucb_score", ascending=False).iloc[0]
    c = candidates[
        (candidates["method"] == best["method"]) &
        (candidates["k"].astype(int) == int(best["k"]))
    ]
    if len(c) == 0:
        return None, {}
    return c.iloc[0], {"predicted_score": float(best["ucb_score"])}


def train_2c_lite_models(train, hazard_positions):
    """
    Train one sklearn regressor per hazard position.

    This is request-level 2C-lite:
      x -> hazard_pos_j

    Not true block-level censored survival.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except Exception as e:
        raise RuntimeError(
            "sklearn is required for 2C-lite. Install scikit-learn or use the bucketed controller only."
        ) from e

    cand = train[train["is_candidate"]].copy()

    feature_cols_num = [
        c for c in [
            "k",
            "temperature",
            "mean_entropy",
            "repetition_density",
        ]
        if c in cand.columns
    ]
    feature_cols_cat = [c for c in ["workload", "method"] if c in cand.columns]

    X_base = cand[feature_cols_num + feature_cols_cat].copy()
    X = pd.get_dummies(X_base, columns=feature_cols_cat, dummy_na=True)
    feature_columns = list(X.columns)

    models = {}
    metrics = []

    for j in hazard_positions:
        ycol = f"hazard_pos_{j}"
        if ycol not in cand.columns:
            continue
        y = safe_num(cand[ycol]).clip(0.0, 1.0)
        mask = y.notna()
        if mask.sum() < 50:
            continue

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=17 + j,
            ),
        )
        model.fit(X.loc[mask], y.loc[mask])
        pred = np.clip(model.predict(X.loc[mask]), 1e-4, 1.0 - 1e-4)
        mae = float(np.mean(np.abs(pred - y.loc[mask].to_numpy())))
        models[j] = model
        metrics.append({"position": j, "train_n": int(mask.sum()), "train_mae": mae})

    meta = {
        "feature_columns": feature_columns,
        "numeric_features": feature_cols_num,
        "categorical_features": feature_cols_cat,
        "train_metrics": metrics,
    }
    return models, meta


def predict_2c_lite(candidates, models, meta, hazard_positions):
    X_base = candidates[meta["numeric_features"] + meta["categorical_features"]].copy()
    X = pd.get_dummies(X_base, columns=meta["categorical_features"], dummy_na=True)
    X = X.reindex(columns=meta["feature_columns"], fill_value=0)

    out = candidates.copy()
    for j in hazard_positions:
        if j in models:
            out[f"pred_hazard_pos_{j}"] = np.clip(models[j].predict(X), 1e-4, 1.0 - 1e-4)
        else:
            out[f"pred_hazard_pos_{j}"] = np.nan

    pred_E = []
    for _, r in out.iterrows():
        k = int(r["k"])
        S = 1.0
        E = 0.0
        for j in range(k):
            col = f"pred_hazard_pos_{j}"
            if col in out.columns and np.isfinite(r[col]):
                h = float(r[col])
            else:
                # If model lacks deeper positions, use conservative stop.
                h = 1.0
            S *= (1.0 - h)
            E += S
        pred_E.append(E)

    out["pred_2c_E_L"] = pred_E
    return out


def select_2c_lite(candidates, cost_tables, models, meta, hazard_positions):
    pred = predict_2c_lite(candidates, models, meta, hazard_positions)

    rows = []
    for idx, r in pred.iterrows():
        # Cost fallback uses no-state bucket table.
        E, C, Sp = predict_bucket_values(r, cost_tables, "all")
        pred_C = C
        if not np.isfinite(pred_C) or pred_C <= 0:
            score = -np.inf
        else:
            score = float(r["pred_2c_E_L"]) / pred_C
        rows.append((idx, score, r["pred_2c_E_L"], pred_C))

    s = pd.DataFrame(rows, columns=["idx", "score", "pred_E_L", "pred_cost_s"])
    s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    if len(s) == 0:
        return None, {}, pred
    best = s.sort_values("score", ascending=False).iloc[0]
    return candidates.loc[best["idx"]], {
        "predicted_score": float(best["score"]),
        "predicted_E_L": float(best["pred_E_L"]),
        "predicted_cost_s": float(best["pred_cost_s"]) if np.isfinite(best["pred_cost_s"]) else np.nan,
    }, pred


def evaluate_seed(df, hazard_positions, seed, train_frac, ucb_alpha):
    train_keys, test_keys = train_test_split_by_request(df, seed, train_frac)
    train_raw = df[df["request_key"].isin(train_keys)].copy()
    test_raw = df[df["request_key"].isin(test_keys)].copy()

    full_b = add_buckets(df, train_raw)
    train = full_b[full_b["request_key"].isin(train_keys)].copy()
    test = full_b[full_b["request_key"].isin(test_keys)].copy()

    tables_full = group_stats(train, "state_full")
    tables_no_entropy = group_stats(train, "state_no_entropy")
    tables_no_repetition = group_stats(train, "state_no_repetition")
    tables_none = group_stats(train, "state_none")

    models_2c, meta_2c = train_2c_lite_models(train, hazard_positions)

    decisions = []
    pred_rows = []

    cand_test = test[test["is_candidate"]].copy()

    for request_key, group in cand_test.groupby("request_key"):
        group = group.copy()
        workload = group["workload"].iloc[0]

        policy_outputs = {}

        # Oracle
        oracle = choose_max(group, "realized_speedup")
        if oracle is not None:
            policy_outputs["oracle_best_per_request"] = (oracle, {})

        # Best fixed method/k per workload
        sel, meta = select_best_fixed(group, tables_full)
        if sel is not None:
            policy_outputs["best_fixed_method_k_per_workload"] = (sel, meta)

        # Entropy/repetition rule
        sel, meta = select_entropy_rule(group, tables_full)
        if sel is not None:
            policy_outputs["entropy_repetition_rule"] = (sel, meta)

        # UCB
        sel, meta = select_ucb(group, tables_full, alpha=ucb_alpha)
        if sel is not None:
            policy_outputs["bandit_ucb_workload"] = (sel, meta)

        # APEX full
        sel, meta = select_bucket_apex(group, tables_full, "state_full", no_survival=False)
        if sel is not None:
            policy_outputs["apex_bucketed_full"] = (sel, meta)

        # no entropy
        sel, meta = select_bucket_apex(group, tables_no_entropy, "state_no_entropy", no_survival=False)
        if sel is not None:
            policy_outputs["apex_no_entropy"] = (sel, meta)

        # no repetition
        sel, meta = select_bucket_apex(group, tables_no_repetition, "state_no_repetition", no_survival=False)
        if sel is not None:
            policy_outputs["apex_no_repetition"] = (sel, meta)

        # no survival/E[L], cost-only
        sel, meta = select_bucket_apex(group, tables_none, "state_none", no_survival=True)
        if sel is not None:
            policy_outputs["apex_no_survival"] = (sel, meta)

        # 2C-lite learned hazard
        sel, meta, pred_group = select_2c_lite(group, tables_none, models_2c, meta_2c, hazard_positions)
        pred_group = pred_group.copy()
        pred_group["seed"] = seed
        pred_group["request_key"] = request_key
        pred_rows.append(pred_group)
        if sel is not None:
            policy_outputs["apex_2c_lite_hazard"] = (sel, meta)

        for policy, (selected, meta) in policy_outputs.items():
            row = {
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
            row.update(meta)
            decisions.append(row)

    dec = pd.DataFrame(decisions)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    return dec, preds, meta_2c, models_2c


def summarize(decisions):
    def p95(x):
        x = safe_num(x).dropna()
        if len(x) == 0:
            return np.nan
        return float(np.nanpercentile(x, 95))

    if len(decisions) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    overall = (
        decisions.groupby("policy")
        .agg(
            n=("realized_speedup", "count"),
            mean_speedup=("realized_speedup", "mean"),
            median_speedup=("realized_speedup", "median"),
            p10_speedup=("realized_speedup", lambda x: float(np.nanpercentile(safe_num(x).dropna(), 10)) if safe_num(x).notna().sum() else np.nan),
            p90_speedup=("realized_speedup", lambda x: float(np.nanpercentile(safe_num(x).dropna(), 90)) if safe_num(x).notna().sum() else np.nan),
            mean_E_L=("realized_E_L", "mean"),
            mean_latency_s=("latency_s", "mean"),
            p95_latency_s=("latency_s", p95),
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
        )
        .reset_index()
    )

    dist = (
        decisions.groupby(["policy", "selected_method", "selected_k"])
        .size()
        .reset_index(name="count")
    )
    dist["fraction_within_policy"] = dist["count"] / dist.groupby("policy")["count"].transform("sum")

    return overall, by_workload, dist


def common_support(decisions, policies):
    d = decisions[decisions["policy"].isin(policies)].copy()
    counts = d.groupby(["seed", "request_key"])["policy"].nunique().reset_index(name="n_policies")
    valid = counts[counts["n_policies"] == len(policies)][["seed", "request_key"]]
    out = d.merge(valid, on=["seed", "request_key"], how="inner")
    return out


def add_oracle_regret(decisions):
    """
    Add oracle_speedup and regret_vs_oracle safely.

    This function may be called multiple times. If oracle_speedup or
    regret_vs_oracle already exists, remove them first to avoid pandas
    merge suffixes such as oracle_speedup_x/oracle_speedup_y.
    """
    if decisions is None or len(decisions) == 0:
        return decisions

    d = decisions.copy()

    # Avoid duplicate columns after repeated calls.
    for col in ["oracle_speedup", "regret_vs_oracle"]:
        if col in d.columns:
            d = d.drop(columns=[col])

    required = {"seed", "request_key", "policy", "realized_speedup"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"add_oracle_regret missing columns: {sorted(missing)}")

    oracle = (
        d[d["policy"] == "oracle_best_per_request"]
        [["seed", "request_key", "realized_speedup"]]
        .rename(columns={"realized_speedup": "oracle_speedup"})
    )

    # If duplicate oracle rows exist, average them.
    oracle = (
        oracle.groupby(["seed", "request_key"], as_index=False)["oracle_speedup"]
        .mean()
    )

    d = d.merge(oracle, on=["seed", "request_key"], how="left")

    if "oracle_speedup" not in d.columns:
        raise RuntimeError(
            "oracle_speedup was not created. Check oracle_best_per_request rows."
        )

    d["regret_vs_oracle"] = d["oracle_speedup"] - d["realized_speedup"]
    return d


def paired_bootstrap(decisions, policy_a, policy_b, n_boot=2000, seed=123):
    """
    Bootstrap paired differences policy_a - policy_b over seed/request pairs.
    """
    d = decisions[decisions["policy"].isin([policy_a, policy_b])].copy()
    pivot = d.pivot_table(
        index=["seed", "request_key", "workload"],
        columns="policy",
        values="realized_speedup",
        aggfunc="mean",
    ).reset_index()

    if policy_a not in pivot.columns or policy_b not in pivot.columns:
        return None

    pivot = pivot.dropna(subset=[policy_a, policy_b]).copy()
    if len(pivot) == 0:
        return None

    diff = pivot[policy_a] - pivot[policy_b]
    obs = float(diff.mean())

    rng = np.random.default_rng(seed)
    vals = diff.to_numpy()
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), len(vals))
        boots.append(float(vals[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_win = float(np.mean(np.array(boots) > 0.0))

    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "n_pairs": int(len(vals)),
        "mean_diff_a_minus_b": obs,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_p_diff_gt_0": p_win,
    }


def make_calibration(preds, hazard_positions, out_dir):
    rows = []
    for j in hazard_positions:
        pcol = f"pred_hazard_pos_{j}"
        ycol = f"hazard_pos_{j}"
        if pcol not in preds.columns or ycol not in preds.columns:
            continue
        sub = preds[[pcol, ycol]].copy()
        sub[pcol] = safe_num(sub[pcol])
        sub[ycol] = safe_num(sub[ycol])
        sub = sub.dropna()
        if len(sub) < 50:
            continue
        try:
            sub["bin"] = pd.qcut(sub[pcol], q=10, duplicates="drop")
        except Exception:
            continue
        cal = (
            sub.groupby("bin", observed=True)
            .agg(
                n=(ycol, "count"),
                pred_mean=(pcol, "mean"),
                obs_mean=(ycol, "mean"),
                mae=(ycol, lambda y: np.nan),
            )
            .reset_index()
        )
        cal["position"] = j
        rows.append(cal)

    cal_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(cal_all):
        cal_all.to_csv(out_dir / "hazard_2c_lite_calibration.csv", index=False)
    return cal_all


def make_figures(out_dir, common_summary, by_workload, dist, cal):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if len(common_summary):
        d = common_summary.sort_values("mean_speedup")
        plt.figure(figsize=(10, max(4, 0.35 * len(d))))
        plt.barh(d["policy"], d["mean_speedup"])
        plt.xlabel("Mean speedup vs AR")
        plt.title("Common-support policy comparison")
        plt.tight_layout()
        plt.savefig(fig_dir / "common_support_policy_mean_speedup.png", dpi=200)
        plt.close()

    main = ["best_fixed_method_k_per_workload", "bandit_ucb_workload", "entropy_repetition_rule", "apex_bucketed_full", "apex_2c_lite_hazard"]
    d = by_workload[by_workload["policy"].isin(main)].copy()
    if len(d):
        pivot = d.pivot_table(index="workload", columns="policy", values="mean_speedup", aggfunc="mean")
        plt.figure(figsize=(12, max(4, 0.45 * len(pivot))))
        im = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(im, label="Mean speedup")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.title("Per-workload policy speedup")
        plt.tight_layout()
        plt.savefig(fig_dir / "per_workload_policy_speedup_heatmap.png", dpi=200)
        plt.close()

    d = dist[dist["policy"].isin(["apex_bucketed_full", "apex_2c_lite_hazard", "entropy_repetition_rule", "bandit_ucb_workload"])].copy()
    if len(d):
        labels = d["selected_method"] + "_k" + d["selected_k"].astype(str)
        d = d.assign(arm=labels)
        pivot = d.pivot_table(index="arm", columns="policy", values="fraction_within_policy", aggfunc="sum").fillna(0)
        plt.figure(figsize=(12, max(4, 0.35 * len(pivot))))
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot.columns))
        # Easier view: grouped bars per policy top arms would be complex; write heatmap.
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(label="Fraction")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.title("Selected method/k distribution")
        plt.tight_layout()
        plt.savefig(fig_dir / "selected_method_k_distribution_heatmap.png", dpi=200)
        plt.close()

    if len(cal):
        for j, sub in cal.groupby("position"):
            plt.figure(figsize=(5, 5))
            plt.scatter(sub["pred_mean"], sub["obs_mean"])
            lo = min(sub["pred_mean"].min(), sub["obs_mean"].min())
            hi = max(sub["pred_mean"].max(), sub["obs_mean"].max())
            plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.xlabel("Predicted hazard")
            plt.ylabel("Observed hazard")
            plt.title(f"2C-lite calibration hazard pos {j}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"hazard_2c_lite_calibration_pos{j}.png", dpi=200)
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--temperature", default="0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--ucb-alpha", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, el_col, hazard_positions = preprocess(Path(args.input), args.temperature)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    all_decisions = []
    all_preds = []
    model_metas = []
    models_by_seed=[]

    for seed in seeds:
        dec, preds, meta, models_2c= evaluate_seed(
            df=df,
            hazard_positions=hazard_positions,
            seed=seed,
            train_frac=args.train_frac,
            ucb_alpha=args.ucb_alpha,
        )
        all_decisions.append(dec)
        all_preds.append(preds)
        model_metas.append({"seed": seed, "meta": meta})
        models_by_seed.append({"seed": seed, "models": models_2c, "meta": meta})
    decisions = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    decisions = add_oracle_regret(decisions)

    overall, by_workload, dist = summarize(decisions)
    decisions.to_csv(out_dir / "decisions_all.csv", index=False)
    overall.to_csv(out_dir / "controller_comparison_full.csv", index=False)
    by_workload.to_csv(out_dir / "per_workload_all_policies.csv", index=False)
    dist.to_csv(out_dir / "selected_method_k_distribution.csv", index=False)

    common_policies = [
        "best_fixed_method_k_per_workload",
        "entropy_repetition_rule",
        "bandit_ucb_workload",
        "apex_bucketed_full",
        "apex_no_entropy",
        "apex_no_repetition",
        "apex_no_survival",
        "apex_2c_lite_hazard",
        "oracle_best_per_request",
    ]
    common = common_support(decisions, common_policies)
    common = add_oracle_regret(common)
    common_summary, common_by_workload, common_dist = summarize(common)
    common.to_csv(out_dir / "decisions_common_support.csv", index=False)
    common_summary.to_csv(out_dir / "controller_comparison_common_support.csv", index=False)
    common_by_workload.to_csv(out_dir / "per_workload_common_support.csv", index=False)

    # Main per-workload table requested.
    main_w = common_by_workload[
        common_by_workload["policy"].isin(
            [
                "apex_bucketed_full",
                "apex_2c_lite_hazard",
                "bandit_ucb_workload",
                "best_fixed_method_k_per_workload",
                "entropy_repetition_rule",
            ]
        )
    ].copy()
    main_w.to_csv(out_dir / "per_workload_apex_vs_ucb_bestfixed.csv", index=False)

    # Ablation summary
    ablation_policies = ["apex_bucketed_full", "apex_no_entropy", "apex_no_repetition", "apex_no_survival", "apex_2c_lite_hazard"]
    ablation = common_summary[common_summary["policy"].isin(ablation_policies)].copy()
    ablation.to_csv(out_dir / "ablation_summary.csv", index=False)

    # Bootstrap CIs
    boot_rows = []
    comparisons = [
        ("apex_bucketed_full", "entropy_repetition_rule"),
        ("apex_bucketed_full", "bandit_ucb_workload"),
        ("apex_bucketed_full", "best_fixed_method_k_per_workload"),
        ("apex_2c_lite_hazard", "apex_bucketed_full"),
        ("apex_2c_lite_hazard", "bandit_ucb_workload"),
        ("apex_2c_lite_hazard", "best_fixed_method_k_per_workload"),
        ("apex_bucketed_full", "apex_no_entropy"),
        ("apex_bucketed_full", "apex_no_repetition"),
        ("apex_bucketed_full", "apex_no_survival"),
    ]
    for a, b in comparisons:
        r = paired_bootstrap(common, a, b, n_boot=args.bootstrap, seed=123)
        if r is not None:
            boot_rows.append(r)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out_dir / "paired_bootstrap_ci.csv", index=False)

    if len(preds):
        preds.to_csv(out_dir / "hazard_2c_lite_predictions.csv", index=False)

    cal = make_calibration(preds, hazard_positions, out_dir)
    make_figures(out_dir, common_summary, main_w, dist, cal)
    save_2c_lite_models(models_by_seed, out_dir)

    diagnostics = {
        "input": args.input,
        "output_dir": str(out_dir),
        "temperature": args.temperature,
        "rows_after_filter": int(len(df)),
        "candidate_rows": int(df["is_candidate"].sum()),
        "expected_length_column": el_col,
        "hazard_positions": hazard_positions,
        "seeds": seeds,
        "train_frac": args.train_frac,
        "ucb_alpha": args.ucb_alpha,
        "common_support_rows": int(len(common)),
        "common_support_requests": int(common["request_key"].nunique()) if len(common) else 0,
        "model_metas": model_metas,
    }
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("Done.")
    print(f"Input: {args.input}")
    print(f"Output: {out_dir}")
    print(f"Rows after filter: {len(df)}")
    print(f"Candidate rows: {int(df['is_candidate'].sum())}")
    print(f"E[L] column: {el_col}")
    print(f"Hazard positions: {hazard_positions}")
    print()
    print("Main files:")
    print(f"  {out_dir / 'controller_comparison_common_support.csv'}")
    print(f"  {out_dir / 'paired_bootstrap_ci.csv'}")
    print(f"  {out_dir / 'per_workload_apex_vs_ucb_bestfixed.csv'}")
    print(f"  {out_dir / 'selected_method_k_distribution.csv'}")
    print(f"  {out_dir / 'ablation_summary.csv'}")
    print(f"  {out_dir / 'hazard_2c_lite_calibration.csv'}")
    print(f"  {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
