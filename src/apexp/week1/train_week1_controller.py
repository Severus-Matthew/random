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
        "method", "k", "temperature", "draft_model", "eagle3_model",
        "ngram_lookup_min", "ngram_lookup_max", "max_prompt_tokens", "num_turns",
    ])

    def one(r):
        method = str(r.get("method", "NA"))
        k = r.get("k", "NA")
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


def safe_mean(x):
    return float(np.nanmean(x)) if len(x) else np.nan


def choose_by_candidate_mean(group, candidate_scores):
    scores = group["candidate_id"].map(candidate_scores).fillna(-1e18)
    if scores.max() <= -1e17:
        return group.loc[group["tokens_per_sec"].idxmax()]
    return group.loc[scores.idxmax()]


def evaluate_choices(test, pred_col, train):
    group_cols = ["workload", "prompt_hash"]
    rows = []

    global_scores = train.groupby("candidate_id")["tokens_per_sec"].mean()
    workload_scores = train.groupby(["workload", "candidate_id"])["tokens_per_sec"].mean()

    for key, g in test.groupby(group_cols):
        workload, prompt_hash = key
        if len(g) < 2:
            continue

        oracle = g.loc[g["tokens_per_sec"].idxmax()]

        controller = g.loc[g[pred_col].idxmax()]

        ar_rows = g[g["method"] == "ar"]
        ar = ar_rows.iloc[0] if len(ar_rows) else None

        global_choice = choose_by_candidate_mean(g, global_scores)

        wl_scores = workload_scores.loc[workload] if workload in workload_scores.index.get_level_values(0) else global_scores
        per_workload_choice = choose_by_candidate_mean(g, wl_scores)

        def pack(name, r):
            ar_tps = float(ar["tokens_per_sec"]) if ar is not None else np.nan
            oracle_tps = float(oracle["tokens_per_sec"])
            tps = float(r["tokens_per_sec"])
            return {
                "workload": workload,
                "prompt_hash": prompt_hash,
                "policy": name,
                "chosen_candidate": r["candidate_id"],
                "chosen_method": r["method"],
                "chosen_k": r["k"],
                "tokens_per_sec": tps,
                "latency_s": float(r["latency_s"]),
                "oracle_tps": oracle_tps,
                "ar_tps": ar_tps,
                "speedup_vs_ar": tps / ar_tps if ar_tps and np.isfinite(ar_tps) and ar_tps > 0 else np.nan,
                "oracle_fraction": tps / oracle_tps if oracle_tps and oracle_tps > 0 else np.nan,
            }

        rows.append(pack("controller", controller))
        rows.append(pack("oracle", oracle))
        rows.append(pack("ar", ar if ar is not None else oracle))
        rows.append(pack("best_global_fixed", global_choice))
        rows.append(pack("best_per_workload_fixed", per_workload_choice))

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--include-state-features", action="store_true",
                    help="Include mean_entropy/repetition_density as state features. Useful as an upper-bound/post-hoc controller.")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "week1" / "controller"
    out_dir.mkdir(parents=True, exist_ok=True)

    req = pd.read_csv(root / "request_summary_all.csv")
    req = to_num(req, [
        "k", "temperature", "max_prompt_tokens", "num_turns",
        "prompt_token_len", "latency_s", "tokens_per_sec",
        "n_output_tokens", "mean_entropy", "repetition_density",
        "acceptance_rate", "draft_tokens", "accepted_tokens_total",
    ])

    req = add_candidate_id(req)
    req = filter_core(req)

    needed = ["workload", "prompt_hash", "method", "k", "candidate_id", "tokens_per_sec", "latency_s"]
    for c in needed:
        if c not in req.columns:
            raise SystemExit(f"Missing required column: {c}")

    data = req.dropna(subset=["prompt_hash", "tokens_per_sec", "latency_s"]).copy()

    # Keep only prompt groups that have at least two candidate choices.
    group_cols = ["workload", "prompt_hash"]
    counts = data.groupby(group_cols)["candidate_id"].nunique().reset_index(name="n_candidates")
    valid = counts[counts["n_candidates"] >= 2][group_cols]
    data = data.merge(valid, on=group_cols, how="inner")

    if len(data) < 20:
        raise SystemExit(f"Not enough controller rows after filtering: {len(data)}")

    data["group_key"] = data["workload"].astype(str) + "::" + data["prompt_hash"].astype(str)

    groups = np.array(sorted(data["group_key"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(groups)

    n_train = max(1, int(0.8 * len(groups)))
    train_groups = set(groups[:n_train])
    test_groups = set(groups[n_train:])

    train = data[data["group_key"].isin(train_groups)].copy()
    test = data[data["group_key"].isin(test_groups)].copy()

    cat_cols = [
        "workload", "method", "candidate_id", "draft_model", "eagle3_model",
        "ngram_lookup_min", "ngram_lookup_max",
    ]
    num_cols = ["k", "temperature", "prompt_token_len", "max_prompt_tokens", "num_turns"]

    if args.include_state_features:
        num_cols += ["mean_entropy", "repetition_density"]

    for c in cat_cols:
        if c not in data.columns:
            data[c] = "NA"
            train[c] = "NA"
            test[c] = "NA"

    for c in num_cols:
        if c not in data.columns:
            data[c] = 0.0
            train[c] = 0.0
            test[c] = 0.0

    model_type = None

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ]
        )

        model = Pipeline([
            ("pre", pre),
            ("rf", RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        model.fit(train[cat_cols + num_cols], train["tokens_per_sec"])
        test["pred_tps"] = model.predict(test[cat_cols + num_cols])
        train["pred_tps"] = model.predict(train[cat_cols + num_cols])

        mae = mean_absolute_error(test["tokens_per_sec"], test["pred_tps"])
        r2 = r2_score(test["tokens_per_sec"], test["pred_tps"])
        model_type = "random_forest"

    except Exception as e:
        # Fallback: use train mean per candidate.
        print("WARNING: sklearn model failed; using candidate-mean fallback:", e)
        means = train.groupby("candidate_id")["tokens_per_sec"].mean()
        global_mean = train["tokens_per_sec"].mean()
        test["pred_tps"] = test["candidate_id"].map(means).fillna(global_mean)
        train["pred_tps"] = train["candidate_id"].map(means).fillna(global_mean)
        mae = float(np.nanmean(np.abs(test["tokens_per_sec"] - test["pred_tps"])))
        r2 = np.nan
        model_type = "candidate_mean_fallback"

    choices = evaluate_choices(test, "pred_tps", train)
    choices.to_csv(out_dir / "controller_choices.csv", index=False)

    summary = choices.groupby("policy").agg(
        n_prompt_groups=("prompt_hash", "nunique"),
        mean_tps=("tokens_per_sec", "mean"),
        median_tps=("tokens_per_sec", "median"),
        mean_latency_s=("latency_s", "mean"),
        mean_speedup_vs_ar=("speedup_vs_ar", "mean"),
        mean_oracle_fraction=("oracle_fraction", "mean"),
    ).reset_index().sort_values("mean_tps", ascending=False)

    summary.to_csv(out_dir / "controller_eval_summary.csv", index=False)

    pred = test[[
        "workload", "prompt_hash", "candidate_id", "method", "k",
        "tokens_per_sec", "pred_tps", "latency_s",
    ]].copy()
    pred.to_csv(out_dir / "controller_pred_vs_actual.csv", index=False)

    metadata = {
        "model_type": model_type,
        "include_state_features": bool(args.include_state_features),
        "n_rows_total": int(len(data)),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_groups": int(len(train_groups)),
        "n_test_groups": int(len(test_groups)),
        "mae_tps": float(mae),
        "r2_tps": float(r2) if np.isfinite(r2) else None,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }

    with open(out_dir / "controller_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Plot comparison.
    if len(summary):
        s = summary.sort_values("mean_tps")
        plt.figure(figsize=(8, 5))
        plt.barh(s["policy"], s["mean_tps"])
        plt.xlabel("Mean actual tokens/sec selected")
        plt.ylabel("Policy")
        plt.title("Week-1 controller vs baselines")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_controller_vs_baselines.png", dpi=180)
        plt.close()

    # Plot controller choices.
    ctrl = choices[choices["policy"] == "controller"]
    if len(ctrl):
        cc = ctrl["chosen_candidate"].value_counts().head(20).iloc[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(cc.index.astype(str), cc.values)
        plt.xlabel("Chosen prompt groups")
        plt.ylabel("Candidate")
        plt.title("Controller choice distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_controller_choice_distribution.png", dpi=180)
        plt.close()

    print("Wrote controller outputs to:", out_dir)
    print(json.dumps(metadata, indent=2))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
