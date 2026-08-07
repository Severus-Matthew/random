#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.apex_eval_utils import ensure_dir, save_plot_metric_by_variant, write_json


PLOT_METRICS = [
    "train_loss",
    "policy_oracle_fraction",
    "policy_exact_k_match_proxy",
    "mae_log_tps",
    "mae_utility",
    "mae_accepted_len",
    "mae_log_cost",
]


def load_variant_history(model_dir: Path, root_name: str) -> pd.DataFrame | None:
    hist_path = model_dir / "training_history.csv"
    if not hist_path.exists():
        return None
    try:
        df = pd.read_csv(hist_path)
    except Exception as e:
        print(f"[warn] failed to read {hist_path}: {e}")
        return None

    df["root_name"] = root_name
    df["variant"] = model_dir.name
    df["model_dir"] = str(model_dir)

    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            for k in [
                "alpha_accept",
                "alpha_waste",
                "alpha_global",
                "alpha_method",
                "lambda_survival",
                "lambda_cost",
                "lambda_tps",
                "lambda_utility",
                "lambda_rank",
                "lr",
                "hidden",
                "dropout",
                "batch_size",
            ]:
                if k in meta:
                    df[k] = meta[k]
        except Exception:
            pass
    return df


def select_best_rows(all_hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_dir, g in all_hist.groupby("model_dir", sort=False):
        gg = g.copy()
        if "policy_oracle_fraction" in gg.columns and gg["policy_oracle_fraction"].notna().any():
            # Higher oracle fraction is better. Tie-break with exact proxy and lower MAE.
            score = (
                pd.to_numeric(gg["policy_oracle_fraction"], errors="coerce").fillna(-1.0)
                + 0.05 * pd.to_numeric(gg.get("policy_exact_k_match_proxy", 0.0), errors="coerce").fillna(0.0)
                - 0.02 * pd.to_numeric(gg.get("mae_log_tps", 0.0), errors="coerce").fillna(0.0)
                - 0.02 * pd.to_numeric(gg.get("mae_utility", 0.0), errors="coerce").fillna(0.0)
            )
            idx = score.idxmax()
        elif "train_loss" in gg.columns:
            idx = pd.to_numeric(gg["train_loss"], errors="coerce").idxmin()
        else:
            idx = gg.index[-1]
        rows.append(gg.loc[idx].to_dict())
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["policy_oracle_fraction", "policy_exact_k_match_proxy"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Model roots containing variant subdirectories")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    plot_dir = ensure_dir(out_dir / "plots")

    histories = []
    for root_s in args.roots:
        root = Path(root_s)
        if not root.exists():
            print(f"[warn] missing root: {root}")
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            df = load_variant_history(d, root.name)
            if df is not None:
                histories.append(df)

    if not histories:
        raise SystemExit("No training_history.csv files found")

    all_hist = pd.concat(histories, ignore_index=True)
    all_hist.to_csv(out_dir / "all_training_history.csv", index=False)

    best = select_best_rows(all_hist)
    best.to_csv(out_dir / "best_model_summary.csv", index=False)
    write_json(out_dir / "best_model_summary.json", best.to_dict(orient="records"))

    # Per-root plots and global plots.
    for metric in PLOT_METRICS:
        if metric not in all_hist.columns:
            continue
        save_plot_metric_by_variant(
            all_hist,
            metric=metric,
            out_path=plot_dir / f"all_{metric}.png",
            x="epoch",
            hue="variant",
        )
        for root_name, g in all_hist.groupby("root_name", dropna=False):
            save_plot_metric_by_variant(
                g,
                metric=metric,
                out_path=plot_dir / f"{root_name}_{metric}.png",
                x="epoch",
                hue="variant",
            )

    # Compact printed table.
    cols = [
        "root_name", "variant", "model_dir", "epoch", "train_loss",
        "policy_oracle_fraction", "policy_exact_k_match_proxy",
        "mae_log_tps", "mae_utility", "mae_accepted_len",
    ]
    cols = [c for c in cols if c in best.columns]
    print(best[cols].to_string(index=False))
    print(f"\nWROTE: {out_dir}")


if __name__ == "__main__":
    main()
