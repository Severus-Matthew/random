#!/usr/bin/env python
"""
make_plots.py

Reads aggregate_by_workload_method.csv (from aggregate_results.py) and
all_runs.csv, then produces a full plot suite:

  1. Throughput & latency bar charts (workload × method)
  2. Acceptance rate & accepted tokens/verifier-pass bars
  3. Rollback frequency & verifier utilisation bars
  4. Structural metrics heatmaps (repetition, entropy, volatility, locality)
  5. Fixed-k sensitivity curves per workload
  6. AR speedup ratio (tps_method / tps_ar) heatmap
  7. Token-regime trace plots (entropy + rejection spikes per example)
  8. Per-workload violin plots of latency distribution
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
METHOD_ORDER  = ["ar", "ngram_sd", "draft_sd", "eagle3"]
METHOD_COLORS = {
    "ar":       "#4C72B0",
    "ngram_sd": "#DD8452",
    "draft_sd": "#55A868",
    "eagle3":   "#C44E52",
}
PALETTE = [METHOD_COLORS.get(m, "#999999") for m in METHOD_ORDER]


def _save(fig, path: Path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ── 1-3. Bar charts ───────────────────────────────────────────────────────
def plot_bars(df: pd.DataFrame, out_dir: Path):
    metrics = [
        ("tps_mean",                              "Throughput (tokens/sec)",              "throughput"),
        ("latency_mean",                          "Mean latency (s)",                     "latency"),
        ("acceptance_rate_mean",                  "Acceptance rate",                      "acceptance_rate"),
        ("accepted_tokens_per_verifier_pass_mean","Accepted tokens / verifier pass",      "accepted_per_vpass"),
        ("rollback_frequency_mean",               "Rollback frequency (rollbacks/vcall)", "rollback_freq"),
        ("verifier_utilization_mean",             "Verifier utilisation",                 "verifier_util"),
    ]

    workloads = sorted(df["workload"].unique())
    methods   = [m for m in METHOD_ORDER if m in df["method"].unique()]

    for col, ylabel, stem in metrics:
        if col not in df.columns:
            continue
        piv = (df.groupby(["workload", "method"])[col]
                 .mean()
                 .unstack("method")
                 .reindex(columns=methods, fill_value=np.nan))

        fig, ax = plt.subplots(figsize=(max(10, len(workloads) * 1.5), 5))
        x = np.arange(len(piv.index))
        w = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            if m not in piv.columns:
                continue
            bars = ax.bar(x + i * w - 0.4 + w / 2,
                          piv[m].values, width=w * 0.9,
                          color=METHOD_COLORS.get(m, "#999"), label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by workload × method")
        ax.legend(title="method")
        _save(fig, out_dir / f"bar_{stem}.png")


# ── 4. Structural heatmaps ────────────────────────────────────────────────
def plot_heatmaps(df: pd.DataFrame, out_dir: Path):
    metrics = [
        ("repetition_density_mean",    "Repetition density",    "RdYlGn"),
        ("entropy_mean",               "Mean entropy",          "YlOrRd"),
        ("acceptance_volatility_mean", "Acceptance volatility", "RdYlGn_r"),
        ("rejection_locality_mean",    "Rejection locality",    "RdYlGn_r"),
        ("rollback_frequency_mean",    "Rollback frequency",    "RdYlGn_r"),
        ("verifier_utilization_mean",  "Verifier utilisation",  "YlGn"),
    ]
    for col, title, cmap in metrics:
        if col not in df.columns:
            continue
        piv = (df.groupby(["workload", "method"])[col]
                 .mean()
                 .unstack("method")
                 .reindex(columns=[m for m in METHOD_ORDER if m in df["method"].unique()]))
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(piv.columns) * 2), max(4, len(piv.index) * 0.8)))
        sns.heatmap(piv, annot=True, fmt=".3f", cmap=cmap, ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{title} — workload × method")
        ax.set_xlabel("method")
        ax.set_ylabel("workload")
        _save(fig, out_dir / f"heatmap_{col}.png")


# ── 5. Fixed-k sensitivity ────────────────────────────────────────────────
def plot_k_sensitivity(df: pd.DataFrame, out_dir: Path):
    k_dir = out_dir / "k_sensitivity"
    k_dir.mkdir(exist_ok=True)

    sweep_metrics = [
        ("tps_mean",               "tokens/sec"),
        ("latency_mean",           "latency (s)"),
        ("acceptance_rate_mean",   "acceptance rate"),
        ("rollback_frequency_mean","rollback freq"),
        ("verifier_utilization_mean","verifier util"),
        ("entropy_mean",           "mean entropy"),
        ("repetition_density_mean","repetition density"),
    ]

    for workload, sub in df.groupby("workload"):
        for col, ylabel in sweep_metrics:
            if col not in sub.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            plotted = False
            for method, mdf in sub.groupby("method"):
                if method == "ar":
                    continue
                mdf = mdf.dropna(subset=["k"]).sort_values("k")
                if mdf.empty or col not in mdf.columns:
                    continue
                ax.plot(mdf["k"], mdf[col],
                        marker="o", label=method,
                        color=METHOD_COLORS.get(method, "#999"))
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_title(f"{workload} — k sensitivity: {col}")
            ax.set_xlabel("k (speculation depth)")
            ax.set_ylabel(ylabel)
            ax.legend()
            _save(fig, k_dir / f"{workload}_{col}.png")


# ── 6. AR speedup heatmap ─────────────────────────────────────────────────
def plot_speedup_heatmap(df: pd.DataFrame, out_dir: Path):
    if "tps_mean" not in df.columns:
        return
    ar_tps = (df[df["method"] == "ar"]
                .groupby("workload")["tps_mean"].mean()
                .rename("ar_tps"))
    merged = df[df["method"] != "ar"].merge(ar_tps, on="workload", how="left")
    merged["speedup"] = merged["tps_mean"] / merged["ar_tps"].replace(0, np.nan)

    piv = (merged.groupby(["workload", "method"])["speedup"]
                 .mean()
                 .unstack("method")
                 .reindex(columns=[m for m in METHOD_ORDER if m != "ar" and m in merged["method"].unique()]))
    if piv.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(piv.columns) * 2.5), max(4, len(piv.index) * 0.9)))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0, ax=ax,
                linewidths=0.5, cbar_kws={"label": "speedup vs AR", "shrink": 0.8})
    ax.set_title("Speedup vs AR (tps_method / tps_ar) — workload × method")
    _save(fig, out_dir / "heatmap_speedup_vs_ar.png")


# ── 7. Token-regime trace plots ───────────────────────────────────────────
def rolling_mean(xs, w=16):
    arr = np.array(xs, dtype=float)
    out = []
    for i in range(len(arr)):
        s = arr[max(0, i - w + 1): i + 1]
        out.append(float(np.nanmean(s)) if np.isfinite(s).any() else np.nan)
    return out


def plot_traces(results_dir: Path, out_dir: Path, max_per_file: int = 2):
    trace_dir = out_dir / "token_regime"
    trace_dir.mkdir(exist_ok=True)

    for trace_file in sorted(results_dir.glob("**/traces.jsonl")):
        count = 0
        for line in open(trace_file):
            rec = json.loads(line)
            entropy  = rec.get("entropy",  [])
            accepted = rec.get("accepted", [])
            rejected = rec.get("rejected", [])
            if not entropy:
                continue
            x = list(range(len(entropy)))

            fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

            # Top: entropy + acceptance rolling mean
            axes[0].plot(x, rolling_mean(entropy, 16),
                         label="entropy (rolling)", color="steelblue", lw=1.4)
            clean_acc = [a for a in accepted if not (isinstance(a, float) and math.isnan(a))]
            if clean_acc:
                axes[0].plot(x, rolling_mean(accepted, 16),
                             label="acceptance (rolling)", color="darkorange", lw=1.4)
            if rejected and sum(rejected) > 0:
                sx = [i for i, r in enumerate(rejected) if r]
                sy = [rolling_mean(entropy, 16)[i] for i in sx]
                axes[0].scatter(sx, sy, label="rejection", color="red", s=14, zorder=5)
            axes[0].set_title(
                f"{rec.get('workload','?')} | {rec.get('method','?')} "
                f"k={rec.get('k','?')} | {rec.get('id','?')} | "
                f"tps={rec.get('tokens_per_sec',float('nan')):.1f}"
            )
            axes[0].set_ylabel("value")
            axes[0].legend(fontsize=8)

            # Bottom: rejection bar
            axes[1].bar(x, rejected if rejected else [0]*len(x),
                        color="red", alpha=0.5, label="rejected token")
            axes[1].set_ylabel("rejection")
            axes[1].set_xlabel("token position")
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            safe_id = str(rec.get("id","0")).replace("/", "_")
            fname = (f"{rec.get('workload','w')}_{rec.get('method','m')}"
                     f"_k{rec.get('k','?')}_{safe_id}.png")
            _save(fig, trace_dir / fname)

            count += 1
            if count >= max_per_file:
                break


# ── 8. Latency violin by workload ─────────────────────────────────────────
def plot_latency_violin(all_df: pd.DataFrame, out_dir: Path):
    if "latency_s" not in all_df.columns:
        return
    methods = [m for m in METHOD_ORDER if m in all_df["method"].unique()]
    for workload, sub in all_df.groupby("workload"):
        sub = sub[sub["method"].isin(methods)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(methods) * 2), 5))
        sns.violinplot(data=sub, x="method", y="latency_s",
                       order=methods,
                       palette=[METHOD_COLORS.get(m, "#999") for m in methods],
                       inner="box", ax=ax)
        ax.set_title(f"{workload} — latency distribution by method")
        ax.set_ylabel("latency (s)")
        _save(fig, out_dir / f"violin_latency_{workload}.png")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir",     default="results/plots")
    ap.add_argument("--trace_examples", type=int, default=2,
                    help="Max token-regime traces per traces.jsonl file")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = results_dir / "aggregate_by_workload_method.csv"
    all_path = results_dir / "all_runs.csv"

    if not agg_path.exists():
        raise SystemExit(f"Run aggregate_results.py first — {agg_path} not found")

    agg = pd.read_csv(agg_path)
    all_df = pd.read_csv(all_path) if all_path.exists() else None

    print("Generating plots ...")

    print("[1/8] Bar charts")
    plot_bars(agg, out_dir)

    print("[2/8] Structural heatmaps")
    plot_heatmaps(agg, out_dir)

    print("[3/8] Fixed-k sensitivity")
    plot_k_sensitivity(agg, out_dir)

    print("[4/8] AR speedup heatmap")
    plot_speedup_heatmap(agg, out_dir)

    print("[5/8] Token-regime traces")
    plot_traces(results_dir, out_dir, max_per_file=args.trace_examples)

    if all_df is not None:
        print("[6/8] Latency violin plots")
        plot_latency_violin(all_df, out_dir)
    else:
        print("[6/8] Skipping violin plots (all_runs.csv not found)")

    print(f"\nAll plots written to {out_dir}")


if __name__ == "__main__":
    main()