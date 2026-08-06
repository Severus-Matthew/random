#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "training_history.csv").exists():
            variants.append(d)

    if not variants:
        raise SystemExit(f"No variant folders with training_history.csv found under {root}")

    summary = []

    histories = {}

    for d in variants:
        name = d.name
        hist = pd.read_csv(d / "training_history.csv")
        histories[name] = hist

        # Choose best epoch by highest policy_oracle_fraction if available.
        if "policy_oracle_fraction" in hist.columns and hist["policy_oracle_fraction"].notna().any():
            best_idx = hist["policy_oracle_fraction"].idxmax()
        else:
            best_idx = hist["train_loss"].idxmin()

        best = hist.loc[best_idx].to_dict()
        best["variant"] = name
        best["best_epoch_index"] = int(best_idx)
        summary.append(best)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "training_curve_summary.csv", index=False)

    print("=== BEST EPOCH SUMMARY ===")
    show_cols = [
        "variant",
        "epoch",
        "train_loss",
        "policy_oracle_fraction",
        "policy_exact_k_match_proxy",
        "mae_accepted_len",
        "mae_log_tps",
        "mae_utility",
    ]
    show_cols = [c for c in show_cols if c in summary_df.columns]
    print(summary_df[show_cols].sort_values("policy_oracle_fraction", ascending=False).to_string(index=False))

    def plot_metric(metric: str, ylabel: str | None = None):
        plt.figure(figsize=(8, 5))
        any_plot = False

        for name, hist in histories.items():
            if metric not in hist.columns:
                continue
            plt.plot(hist["epoch"], hist[metric], marker="o", label=name)
            any_plot = True

        if not any_plot:
            plt.close()
            return

        plt.xlabel("Epoch")
        plt.ylabel(ylabel or metric)
        plt.title(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = out_dir / f"{metric}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print("wrote", out)

    plot_metric("train_loss", "Train loss")
    plot_metric("policy_oracle_fraction", "Policy oracle fraction")
    plot_metric("policy_exact_k_match_proxy", "Exact k match proxy")
    plot_metric("mae_accepted_len", "MAE accepted length")
    plot_metric("mae_log_tps", "MAE log TPS")
    plot_metric("mae_utility", "MAE utility")

    # Combined dashboard-style CSV with all curves.
    merged = []
    for name, hist in histories.items():
        tmp = hist.copy()
        tmp["variant"] = name
        merged.append(tmp)
    pd.concat(merged, ignore_index=True).to_csv(out_dir / "all_training_histories_long.csv", index=False)

    print("wrote", out_dir / "training_curve_summary.csv")
    print("wrote", out_dir / "all_training_histories_long.csv")


if __name__ == "__main__":
    main()
