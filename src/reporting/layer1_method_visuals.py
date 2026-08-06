#!/usr/bin/env python3
"""
Method-stratified Layer-1 visualizations for APEX.

This redraws Layer-1 results so the evidence is not averaged across different
speculative mechanisms. In particular, binned trend plots are computed
separately for ngram_sd, eagle3, and draft_sd.

Inputs:
  --layer1-dir results/apex_layer1

Expected files:
  layer1_request_survival.csv
  layer1_group_survival.csv

Outputs:
  <output-dir>/method_visual_diagnostics.txt
  <output-dir>/method_binned_stats.csv
  <output-dir>/method_best_speedup.csv
  <output-dir>/figures/*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_ORDER = ["ngram_sd", "eagle3", "draft_sd"]
K_ORDER = [1, 2, 4, 8, 16]
PREDICTORS = [
    ("acceptance_rate", "Acceptance rate"),
    ("accepted_per_draft", "Accepted tokens per draft"),
    ("apex_expected_len_survival_observed", "Observed survival-sum E[L]"),
    ("apex_expected_len_counter", "Counter E[L] / accepted per draft"),
    ("mean_entropy", "Mean entropy"),
    ("repetition_density", "Repetition density"),
]


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def finite_rows(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    mask = np.ones(len(out), dtype=bool)
    for c in cols:
        if c not in out.columns:
            return out.iloc[0:0].copy()
        out[c] = to_num(out[c])
        mask &= np.isfinite(out[c].to_numpy(dtype=float, na_value=np.nan))
    return out.loc[mask].copy()


def non_ar_with_survival(req: pd.DataFrame) -> pd.DataFrame:
    out = req[req["method"].astype(str).str.lower() != "ar"].copy()
    if "apex_observed_positions" in out.columns:
        out = out[to_num(out["apex_observed_positions"]).fillna(0) > 0].copy()
    return out


def filter_temperature(df: pd.DataFrame, temperature: str | None) -> pd.DataFrame:
    if temperature is None or temperature == "all" or "temperature" not in df.columns:
        return df.copy()
    t = to_num(df["temperature"])
    try:
        target = float(temperature)
    except ValueError:
        return df.copy()
    return df[np.isclose(t.fillna(target), target)].copy()


def method_sort_key(m: str) -> int:
    try:
        return METHOD_ORDER.index(str(m))
    except ValueError:
        return 99


def k_sort_key(k) -> int:
    try:
        return K_ORDER.index(int(float(k)))
    except Exception:
        return 99


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


def method_stratified_binned_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    out_png: Path,
    out_stats_rows: list[dict],
    n_bins: int = 8,
    min_bin_n: int = 8,
) -> None:
    sub = finite_rows(df, [x_col, y_col])
    sub = sub[sub["method"].isin(METHOD_ORDER)].copy()
    if len(sub) < 30:
        return

    # Common global quantile bins make method curves comparable on the same x-axis.
    try:
        _, edges = pd.qcut(sub[x_col], q=min(n_bins, max(3, len(sub)//200)), retbins=True, duplicates="drop")
    except ValueError:
        return
    if len(edges) < 3:
        return
    edges = np.unique(edges)
    sub["_bin"] = pd.cut(sub[x_col], bins=edges, include_lowest=True, duplicates="drop")

    plt.figure(figsize=(8.0, 5.2))
    any_line = False
    for method in METHOD_ORDER:
        mdf = sub[sub["method"] == method].copy()
        if mdf.empty:
            continue
        g = mdf.groupby("_bin", observed=True).agg(
            x_mid=(x_col, "median"),
            y_med=(y_col, "median"),
            y_q25=(y_col, lambda s: s.quantile(0.25)),
            y_q75=(y_col, lambda s: s.quantile(0.75)),
            n=(y_col, "size"),
        ).reset_index()
        g = g[g["n"] >= min_bin_n].copy()
        if g.empty:
            continue
        any_line = True
        plt.plot(g["x_mid"], g["y_med"], marker="o", label=f"{method}")
        # Keep interval subtle. For many lines, interval may still be visually busy,
        # so only show it when there are <=3 lines, which is true here.
        plt.fill_between(g["x_mid"].to_numpy(), g["y_q25"].to_numpy(), g["y_q75"].to_numpy(), alpha=0.12)
        for _, r in g.iterrows():
            out_stats_rows.append({
                "predictor": x_col,
                "target": y_col,
                "method": method,
                "x_mid": float(r["x_mid"]),
                "y_median": float(r["y_med"]),
                "y_q25": float(r["y_q25"]),
                "y_q75": float(r["y_q75"]),
                "n": int(r["n"]),
            })

    if not any_line:
        plt.close()
        return
    plt.axhline(1.0, linewidth=1, linestyle="--", alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel("Median speedup vs AR")
    plt.title(f"Speedup vs {x_label}, stratified by method")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Method", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close()


def small_multiples_binned_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    out_png: Path,
    n_bins: int = 8,
    min_bin_n: int = 8,
) -> None:
    sub = finite_rows(df, [x_col, y_col])
    sub = sub[sub["method"].isin(METHOD_ORDER)].copy()
    if len(sub) < 30:
        return
    try:
        _, edges = pd.qcut(sub[x_col], q=min(n_bins, max(3, len(sub)//200)), retbins=True, duplicates="drop")
    except ValueError:
        return
    edges = np.unique(edges)
    if len(edges) < 3:
        return
    sub["_bin"] = pd.cut(sub[x_col], bins=edges, include_lowest=True, duplicates="drop")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    plotted = False
    for ax, method in zip(axes, METHOD_ORDER):
        mdf = sub[sub["method"] == method].copy()
        g = mdf.groupby("_bin", observed=True).agg(
            x_mid=(x_col, "median"),
            y_med=(y_col, "median"),
            y_q25=(y_col, lambda s: s.quantile(0.25)),
            y_q75=(y_col, lambda s: s.quantile(0.75)),
            n=(y_col, "size"),
        ).reset_index()
        g = g[g["n"] >= min_bin_n].copy()
        ax.set_title(method)
        ax.axhline(1.0, linewidth=1, linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.3)
        if not g.empty:
            plotted = True
            ax.plot(g["x_mid"], g["y_med"], marker="o")
            ax.fill_between(g["x_mid"].to_numpy(), g["y_q25"].to_numpy(), g["y_q75"].to_numpy(), alpha=0.15)
        ax.set_xlabel(x_label)
    axes[0].set_ylabel("Median speedup vs AR")
    fig.suptitle(f"Speedup vs {x_label}: separate medians per method", y=1.04)
    fig.tight_layout()
    if plotted:
        fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def heatmap(table: pd.DataFrame, out: Path, title: str, colorbar_label: str, fmt: str = ".2f", annotate: bool = True) -> None:
    if table.empty:
        return
    data = table.to_numpy(dtype=float)
    fig_w = max(6.5, 0.9 * table.shape[1] + 3)
    fig_h = max(4.0, 0.45 * table.shape[0] + 2)
    plt.figure(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(data)
    im = plt.imshow(masked, aspect="auto")
    plt.colorbar(im, label=colorbar_label)
    plt.xticks(range(table.shape[1]), table.columns, rotation=0)
    plt.yticks(range(table.shape[0]), table.index)
    plt.title(title)
    if annotate and table.shape[0] * table.shape[1] <= 120:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7)
                else:
                    plt.text(j, i, "—", ha="center", va="center", fontsize=7, alpha=0.55)
    plt.tight_layout()
    plt.savefig(out, dpi=240)
    plt.close()


def method_depth_heatmaps(req: pd.DataFrame, out_dir: Path, temperature: str | None) -> None:
    df = non_ar_with_survival(req)
    df = filter_temperature(df, temperature)
    df = finite_rows(df, ["speedup_vs_ar", "k"])
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    if df.empty:
        return
    df["k"] = to_num(df["k"]).round().astype(int)

    # One heatmap per method: rows workload, columns k. This avoids the huge sparse all-method grid.
    for method in METHOD_ORDER:
        mdf = df[df["method"] == method].copy()
        if mdf.empty:
            continue
        table = mdf.pivot_table(index="workload", columns="k", values="speedup_vs_ar", aggfunc="median")
        table = table.reindex(columns=[k for k in K_ORDER if k in table.columns])
        table = table.sort_index()
        heatmap(table, out_dir / f"method_speedup_heatmap_{method}.png", f"Median speedup vs AR for {method}", "Median speedup", fmt=".2f")

        counts = mdf.pivot_table(index="workload", columns="k", values="speedup_vs_ar", aggfunc="size")
        counts = counts.reindex(index=table.index, columns=table.columns)
        heatmap(counts, out_dir / f"method_coverage_heatmap_{method}.png", f"Run coverage for {method}", "Request count", fmt=".0f")


def common_k4_heatmap(req: pd.DataFrame, out_dir: Path, temperature: str | None) -> None:
    df = non_ar_with_survival(req)
    df = filter_temperature(df, temperature)
    df = finite_rows(df, ["speedup_vs_ar", "k"])
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    if df.empty:
        return
    df["k"] = to_num(df["k"]).round().astype(int)
    k4 = df[df["k"] == 4].copy()
    if k4.empty:
        return
    table = k4.pivot_table(index="workload", columns="method", values="speedup_vs_ar", aggfunc="median")
    table = table.reindex(columns=[m for m in METHOD_ORDER if m in table.columns]).sort_index()
    heatmap(table, out_dir / "common_k4_speedup_by_workload_method.png", "Median speedup at common depth k=4", "Median speedup", fmt=".2f")

    if "apex_expected_len_counter" in k4.columns:
        k4e = finite_rows(k4, ["apex_expected_len_counter"])
        table2 = k4e.pivot_table(index="workload", columns="method", values="apex_expected_len_counter", aggfunc="median")
        table2 = table2.reindex(columns=[m for m in METHOD_ORDER if m in table2.columns]).sort_index()
        heatmap(table2, out_dir / "common_k4_accepted_len_by_workload_method.png", "Median accepted tokens per draft at k=4", "Median accepted tokens", fmt=".2f")


def best_speedup_summary(req: pd.DataFrame, out_dir: Path, temperature: str | None) -> pd.DataFrame:
    df = non_ar_with_survival(req)
    df = filter_temperature(df, temperature)
    df = finite_rows(df, ["speedup_vs_ar", "k"])
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    if df.empty:
        return pd.DataFrame()
    df["k"] = to_num(df["k"]).round().astype(int)
    g = df.groupby(["workload", "method", "k"], as_index=False).agg(
        median_speedup=("speedup_vs_ar", "median"),
        median_tokens_per_sec=("tokens_per_sec", "median") if "tokens_per_sec" in df.columns else ("speedup_vs_ar", "size"),
        n=("speedup_vs_ar", "size"),
    )
    # Pick best observed k per workload/method.
    idx = g.groupby(["workload", "method"])["median_speedup"].idxmax()
    best = g.loc[idx].copy().sort_values(["workload", "method"])
    best.to_csv(out_dir / "method_best_speedup.csv", index=False)

    table = best.pivot_table(index="workload", columns="method", values="median_speedup", aggfunc="median")
    table = table.reindex(columns=[m for m in METHOD_ORDER if m in table.columns]).sort_index()
    heatmap(table, out_dir / "figures" / "best_observed_speedup_by_workload_method.png", "Best observed speedup over k for each method", "Best median speedup", fmt=".2f")

    # Annotated table with k values instead of just speedup.
    ktable = best.pivot_table(index="workload", columns="method", values="k", aggfunc="first")
    ktable = ktable.reindex(columns=[m for m in METHOD_ORDER if m in ktable.columns]).sort_index()
    heatmap(ktable, out_dir / "figures" / "best_observed_k_by_workload_method.png", "Best observed k for each workload/method", "k", fmt=".0f")
    return best


def survival_method_heatmaps(req: pd.DataFrame, out_dir: Path, temperature: str | None) -> None:
    df = non_ar_with_survival(req)
    df = filter_temperature(df, temperature)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    surv_cols = sorted([c for c in df.columns if c.startswith("survival_pos_")], key=lambda c: int(c.rsplit("_", 1)[1]))
    if not surv_cols:
        return
    df = finite_rows(df, ["k"])
    df["k"] = to_num(df["k"]).round().astype(int)
    for method in METHOD_ORDER:
        mdf = df[df["method"] == method].copy()
        if mdf.empty:
            continue
        for c in surv_cols:
            mdf[c] = to_num(mdf[c])
        table = mdf.groupby("k")[surv_cols].median().reindex([k for k in K_ORDER if k in mdf["k"].unique()])
        table.columns = ["S" + c.rsplit("_", 1)[1] for c in table.columns]
        heatmap(table, out_dir / f"method_survival_by_depth_{method}.png", f"Median survival by depth for {method}", "Median survival", fmt=".2f")


def write_diagnostics(req: pd.DataFrame, out_dir: Path, temperature: str | None) -> None:
    df_all = non_ar_with_survival(req)
    df = filter_temperature(df_all, temperature)
    lines = []
    lines.append(f"rows_total: {len(req)}")
    lines.append(f"non_ar_rows_with_survival_all_temperatures: {len(df_all)}")
    lines.append(f"temperature_filter: {temperature}")
    lines.append(f"rows_after_temperature_filter: {len(df)}")
    lines.append("")
    lines.append("rows by method after filter:")
    for m, n in df["method"].value_counts().sort_index().items():
        lines.append(f"  {m}: {n}")
    lines.append("")
    if "k" in df.columns:
        lines.append("coverage by method/k after filter:")
        cov = df.assign(k_round=to_num(df["k"]).round()).groupby(["method", "k_round"]).size()
        for (m, k), n in cov.items():
            lines.append(f"  {m}@k{int(k)}: {n}")
    lines.append("")
    lines.append("Why blanks appear in workload/depth heatmaps:")
    lines.append("  A blank means no request rows exist for that workload/method/k after filtering, not speedup=0.")
    lines.append("  In these results, k=4 is broadly covered; several k=1/2/8/16 sweeps exist only for a subset of workloads/methods.")
    lines.append("  If temperature_filter=0, any T>0 rows are intentionally excluded, which can also create blanks.")
    (out_dir / "method_visual_diagnostics.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer1-dir", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--temperature", default="0", help="Use 0 for T=0 rows, all for all temperatures. Default: 0")
    ap.add_argument("--min-bin-n", type=int, default=8)
    args = ap.parse_args()

    layer1 = Path(args.layer1_dir)
    out_dir = Path(args.output_dir) if args.output_dir else layer1 / "method_clean"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    req_path = layer1 / "layer1_request_survival.csv"
    if not req_path.exists():
        raise FileNotFoundError(req_path)
    req = pd.read_csv(req_path, low_memory=False)

    write_diagnostics(req, out_dir, args.temperature)
    df = non_ar_with_survival(req)
    df = filter_temperature(df, args.temperature)
    df = df[df["method"].isin(METHOD_ORDER)].copy()

    binned_rows: list[dict] = []
    for pred, label in PREDICTORS:
        if pred not in df.columns:
            continue
        method_stratified_binned_plot(
            df, pred, "speedup_vs_ar", label,
            fig_dir / f"method_binned_{safe_name(pred)}_vs_speedup.png",
            binned_rows,
            min_bin_n=args.min_bin_n,
        )
        small_multiples_binned_plot(
            df, pred, "speedup_vs_ar", label,
            fig_dir / f"method_small_multiples_{safe_name(pred)}_vs_speedup.png",
            min_bin_n=args.min_bin_n,
        )
    pd.DataFrame(binned_rows).to_csv(out_dir / "method_binned_stats.csv", index=False)

    method_depth_heatmaps(req, fig_dir, args.temperature)
    common_k4_heatmap(req, fig_dir, args.temperature)
    best_speedup_summary(req, out_dir, args.temperature)
    survival_method_heatmaps(req, fig_dir, args.temperature)

    print(f"Wrote method-stratified Layer-1 visuals to {out_dir}")


if __name__ == "__main__":
    main()
