#!/usr/bin/env python
"""
make_plots.py — Phase 1 + Phase 2 complete plot suite
======================================================

Fixes applied vs previous version:
  - Draft-SD included in ALL plots where it has data (k-sweep, baseline)
  - Structural heatmaps show ALL methods (not AR-only)
  - Draft model sweep handles single-model case + adds AR baseline from main agg
  - Temperature/context/multi-turn explicitly note draft_sd was excluded from those
    experiment configs and label plots accordingly
  - speedup_vs_ar for draft model sweep computed by merging from aggregate_k_sweep
  - acc_vol_mean near-zero values treated as NaN (instrument artifact)
"""
import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 180, "font.family": "DejaVu Sans"})

WORKLOAD_SHORT = {
    "code_gen":                      "Code Gen",
    "conversational_generation":     "Conversational",
    "hardware_gen":                  "HW Gen",
    "long_chain_reasoning":          "Long-Chain",
    "long_context_completion":       "Long-Context",
    "long_horizon_swe":              "SWE",
    "mathematical_reasoning":        "Math",
    "conversational_generation_gen": "Conv-Gen",
    "conversational_generation_sft": "Conv-SFT",
}
METHOD_ORDER  = ["ar", "ngram_sd", "draft_sd", "eagle3"]
METHOD_LABELS = {"ar":"AR","ngram_sd":"Ngram-SD","draft_sd":"Draft-SD","eagle3":"Eagle3"}
COLORS = {"ar":"#4C72B0","ngram_sd":"#DD8452","draft_sd":"#55A868","eagle3":"#C44E52"}


def wl(w): return WORKLOAD_SHORT.get(w, w)
def ml(m): return METHOD_LABELS.get(m, m)
def methods_in(df): return [m for m in METHOD_ORDER if m in df["method"].unique()]
def workloads_in(df): return [w for w in WORKLOAD_SHORT if w in df["workload"].unique()]

def ar_tps_series(df):
    return df[df["method"]=="ar"].groupby("workload")["tps_mean"].mean()

def clean_acc_vol(df):
    """acc_vol near 0 is an instrument artifact (NaN propagation) — replace with NaN."""
    if "acc_vol_mean" in df.columns:
        df = df.copy()
        df.loc[df["acc_vol_mean"].abs() < 1e-10, "acc_vol_mean"] = np.nan
    return df

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")

def rp(rd, name):
    p = Path(rd) / name
    return pd.read_csv(p) if p.exists() else None

def handles_legend(methods):
    return [mpatches.Patch(color=COLORS[m], label=ml(m)) for m in methods]


# ═══════════════════════════════════════════════════════════════
# PHASE 1
# ═══════════════════════════════════════════════════════════════

def fig01_throughput_bar(agg, out):
    """Throughput bar — all 4 methods at k=4 (AR has only k=1 so use its k=1)."""
    # For AR use all k (it only has k=1); for others use k=4
    ar_rows = agg[agg["method"] == "ar"].copy()
    spec    = agg[(agg["method"] != "ar") & (agg["k"] == 4)].copy()
    plot_df = pd.concat([ar_rows, spec])

    workloads = workloads_in(plot_df)
    methods   = methods_in(plot_df)
    x, wb    = np.arange(len(workloads)), 0.18
    n        = len(methods)
    offs     = np.linspace(-(n-1)/2, (n-1)/2, n)

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, m in enumerate(methods):
        sub   = plot_df[plot_df["method"] == m]
        means = [sub[sub["workload"]==w]["tps_mean"].mean() for w in workloads]
        stds  = [sub[sub["workload"]==w]["tps_std"].mean()  for w in workloads]
        ax.bar(x + offs[i]*wb, means, width=wb*0.9,
               color=COLORS[m], label=ml(m),
               yerr=stds, capsize=3, error_kw={"elinewidth":1,"alpha":0.6})

    ax.set_xticks(x)
    ax.set_xticklabels([wl(w) for w in workloads], rotation=20, ha="right")
    ax.set_ylabel("Tokens / Second")
    ax.set_title("Fig 1: Throughput (k=4) — All 4 Methods × All Workloads", fontweight="bold")
    ax.legend(title="Method"); ax.yaxis.grid(True, alpha=0.4)
    save(fig, out/"fig01_throughput_bar_k4.png")


def fig02_speedup_heatmap(agg, out):
    """Peak speedup heatmap — best k per method (all 3 speculative methods)."""
    ar   = ar_tps_series(agg)
    spec = [m for m in ["ngram_sd","draft_sd","eagle3"] if m in agg["method"].unique()]
    wls  = workloads_in(agg)

    best = (agg[agg["method"].isin(spec)]
            .groupby(["workload","method"])["tps_mean"].max()
            .unstack()[spec])
    speedup = best.div(ar, axis=0).reindex(wls)
    speedup.index   = speedup.index.map(wl)
    speedup.columns = [ml(m) for m in spec]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(speedup, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0,
                linewidths=0.5, ax=ax, cbar_kws={"label":"Speedup vs AR"})
    ax.set_title("Fig 2: Peak Speedup vs AR (Best k) — All 3 Speculative Methods",
                 fontweight="bold")
    save(fig, out/"fig02_speedup_heatmap.png")


def fig03_k_sensitivity(agg, out):
    """k-sensitivity for all 3 speculative methods side by side."""
    ks  = sorted(agg["k"].dropna().unique())
    wls = workloads_in(agg)
    pal = sns.color_palette("tab10", len(wls))

    spec_methods = [m for m in ["eagle3","ngram_sd","draft_sd"] if m in agg["method"].unique()]
    fig, axes = plt.subplots(1, len(spec_methods), figsize=(6*len(spec_methods), 5))
    if len(spec_methods) == 1: axes = [axes]

    for ax, meth in zip(axes, spec_methods):
        sub = agg[agg["method"] == meth]
        for j, wl_ in enumerate(wls):
            s = sub[sub["workload"]==wl_].groupby("k")["tps_mean"].mean()
            if s.empty: continue
            ax.plot(s.index, s.values, marker="o", color=pal[j],
                    label=wl(wl_), linewidth=2)
        ax.set_title(f"{ml(meth)} — k Sensitivity", fontweight="bold")
        ax.set_xlabel("k"); ax.set_ylabel("tok/s")
        ax.set_xticks(ks); ax.legend(fontsize=7, ncol=2)
        ax.yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 3: k-Sensitivity — All Speculative Methods", fontweight="bold", y=1.01)
    save(fig, out/"fig03_k_sensitivity.png")


def fig04_latency_cdf(all_df, out):
    """Latency CDF per workload — all 4 methods at k=4."""
    wls   = workloads_in(all_df)
    ncols = 4
    nrows = math.ceil((len(wls)+1)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*3.5))
    axes = axes.flatten()

    for idx, wl_ in enumerate(wls):
        ax = axes[idx]
        for m in METHOD_ORDER:
            sub = all_df[(all_df["workload"]==wl_) & (all_df["method"]==m)]
            if m != "ar": sub = sub[sub["k"]==4]
            vals = np.sort(sub["latency_s"].dropna().values)
            if len(vals) == 0: continue
            ax.plot(vals, np.arange(1,len(vals)+1)/len(vals),
                    color=COLORS[m], label=ml(m), linewidth=1.8)
        ax.set_title(wl(wl_), fontsize=10, fontweight="bold")
        ax.set_xlabel("Latency (s)", fontsize=8)
        ax.set_ylabel("CDF", fontsize=8)
        ax.tick_params(labelsize=8); ax.yaxis.grid(True, alpha=0.4)

    for ax in axes[len(wls):]: ax.axis("off")
    axes[len(wls)].legend(handles=handles_legend(METHOD_ORDER),
                           loc="center", fontsize=11, title="Method")
    fig.suptitle("Fig 4: Latency CDF by Workload — All 4 Methods (k=4)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, out/"fig04_latency_cdf.png")


def fig05_structural_heatmaps(all_df, out):
    """
    Structural heatmaps — ALL methods, not just AR.
    Shows that entropy/repetition density are intrinsic to workload
    (method rows should be nearly identical = method invariance confirmed).
    """
    all_df = clean_acc_vol(all_df)
    metrics = [
        ("mean_entropy",       "Token Entropy",         "YlOrRd", False),
        ("repetition_density", "Repetition Density",    "YlGn",   True),
        ("acceptance_volatility","Acceptance Volatility","Blues",  True),
    ]
    for col, title, cmap, high_good in metrics:
        if col not in all_df.columns: continue
        k4 = all_df.copy()
        k4 = k4[(k4["method"]!="ar") | (k4["k"].isna()) | (k4["k"]==1)]
        piv = (k4.groupby(["workload","method"])[col].mean()
                 .unstack()
                 .reindex(columns=[m for m in METHOD_ORDER if m in k4["method"].unique()])
                 .round(3))
        piv.index = piv.index.map(wl)
        piv.columns = [ml(m) for m in piv.columns]
        if piv.empty or piv.isna().all().all(): continue

        fig, ax = plt.subplots(figsize=(max(8,len(piv.columns)*2.5), max(5,len(piv.index)*0.8)))
        sns.heatmap(piv, annot=True, fmt=".3f", cmap=cmap, ax=ax,
                    linewidths=0.5, cbar_kws={"label":title,"shrink":0.85})
        ax.set_title(f"Fig 5: {title} — Workload × Method\n"
                     f"(Near-identical columns confirm method invariance)",
                     fontweight="bold")
        ax.set_xlabel("Method"); ax.set_ylabel("")
        save(fig, out/f"fig05_structural_{col}.png")


def fig06_entropy_speedup_scatter(agg, out):
    """Entropy/rep-density vs speedup — all speculative methods."""
    ar   = ar_tps_series(agg)
    wls  = workloads_in(agg)
    pal  = {w:sns.color_palette("tab10",len(wls))[i] for i,w in enumerate(wls)}

    entr = agg[agg["method"]=="ar"].groupby("workload")["entropy_mean"].mean()
    rep  = agg[agg["method"]=="ar"].groupby("workload")["rep_density_mean"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, x_data, x_label, meth in [
        (axes[0], entr, "Mean Token Entropy",      "eagle3"),
        (axes[1], rep,  "Repetition Density",       "ngram_sd"),
    ]:
        tps_m   = agg[(agg["method"]==meth)&(agg["k"]==4)].groupby("workload")["tps_mean"].mean()
        speedup = tps_m / ar
        for w in wls:
            if w not in x_data or w not in speedup: continue
            ax.scatter(x_data[w], speedup[w], color=pal[w], s=120, zorder=5)
            ax.annotate(wl(w),(x_data[w],speedup[w]),
                        xytext=(5,5),textcoords="offset points",fontsize=8)
        x_v = np.array([x_data[w] for w in wls if w in speedup and w in x_data])
        y_v = np.array([speedup[w] for w in wls if w in speedup and w in x_data])
        if len(x_v)>1:
            z  = np.polyfit(x_v, y_v, 1)
            xl = np.linspace(x_v.min(), x_v.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), "k--", alpha=0.5, linewidth=1.5)
            r  = np.corrcoef(x_v, y_v)[0,1]
            ax.set_title(f"{ml(meth)} Speedup vs {x_label}\nr = {r:.3f}", fontweight="bold")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel(x_label); ax.set_ylabel("Speedup vs AR (k=4)")
        ax.yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 6: Token Regime → Speedup Relationship", fontweight="bold")
    save(fig, out/"fig06_entropy_speedup_scatter.png")


def fig07_draft_sd_analysis(agg, out):
    """Draft-SD: comparison bar + k-sensitivity showing degradation."""
    ar   = ar_tps_series(agg)
    wls  = workloads_in(agg)
    pal  = sns.color_palette("tab10", len(wls))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — bar at k=4 (all 4 methods)
    ar_rows = agg[agg["method"]=="ar"]; spec = agg[(agg["method"]!="ar")&(agg["k"]==4)]
    plot_df = pd.concat([ar_rows, spec])
    x, wb = np.arange(len(wls)), 0.18
    for i, m in enumerate(methods_in(plot_df)):
        sub   = plot_df[plot_df["method"]==m]
        means = [sub[sub["workload"]==w]["tps_mean"].mean() for w in wls]
        axes[0].bar(x+(i-1.5)*wb, means, width=wb*0.9, color=COLORS[m], label=ml(m))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([wl(w) for w in wls], rotation=20, ha="right")
    axes[0].set_ylabel("tok/s")
    axes[0].set_title("All Methods at k=4\n(Draft-SD uses Qwen2.5-1.5B mismatch)",
                       fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].yaxis.grid(True, alpha=0.4)

    # Right — Draft-SD k-sweep + AR reference
    draft_k = (agg[agg["method"]=="draft_sd"]
               .groupby(["workload","k"])["tps_mean"].mean().unstack())
    for i, w in enumerate(wls):
        if w not in draft_k.index: continue
        axes[1].plot(draft_k.columns, draft_k.loc[w],
                     marker="o", color=pal[i], label=wl(w))
        axes[1].axhline(ar.get(w,np.nan), color=pal[i], linestyle="--", alpha=0.3)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("tok/s")
    axes[1].set_title("Draft-SD k-Sensitivity\n(dashed = AR baseline, all below AR)",
                       fontweight="bold")
    axes[1].legend(fontsize=7, ncol=2); axes[1].yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 7: Draft-SD Performance — Family Mismatch Penalty",
                 fontweight="bold")
    save(fig, out/"fig07_draft_sd_analysis.png")


def fig08_ngram_beats_eagle3(agg, out):
    """Ngram > Eagle3 on long-output workloads."""
    ar   = ar_tps_series(agg)
    long = [w for w in ["long_chain_reasoning","long_context_completion","long_horizon_swe"]
            if w in agg["workload"].unique()]
    ks   = sorted(agg["k"].dropna().unique())

    n  = max(len(long), 1)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    for ax, wl_ in zip(axes, long):
        for m in ["ngram_sd","eagle3","draft_sd"]:
            if m not in agg["method"].unique(): continue
            s = agg[(agg["workload"]==wl_)&(agg["method"]==m)].groupby("k")["tps_mean"].mean()
            if s.empty: continue
            ax.plot(s.index, s.values, marker="o", color=COLORS[m],
                    label=ml(m), linewidth=2.5)
        ax.axhline(ar.get(wl_,np.nan), color=COLORS["ar"], linestyle="--",
                   linewidth=2, label="AR")
        ax.set_title(wl(wl_), fontweight="bold")
        ax.set_xlabel("k"); ax.set_ylabel("tok/s")
        ax.set_xticks(ks); ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 8: Ngram-SD vs Eagle3 vs Draft-SD on Long-Output Workloads",
                 fontweight="bold", y=1.01)
    save(fig, out/"fig08_ngram_beats_eagle3.png")


def fig09_throughput_scaling(agg, out):
    """Full throughput scaling — all workloads, all methods."""
    ar   = ar_tps_series(agg)
    wls  = workloads_in(agg)
    ncols = 4; nrows = math.ceil((len(wls)+1)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows*4))
    axes = axes.flatten()

    for idx, wl_ in enumerate(wls):
        ax = axes[idx]
        for m in ["eagle3","ngram_sd","draft_sd"]:
            if m not in agg["method"].unique(): continue
            s = agg[(agg["workload"]==wl_)&(agg["method"]==m)].groupby("k")["tps_mean"].mean()
            if s.empty: continue
            ax.plot(s.index, s.values, marker="o", color=COLORS[m],
                    label=ml(m), linewidth=2)
        ax.axhline(ar.get(wl_,np.nan), color=COLORS["ar"], linestyle="--",
                   linewidth=2, label="AR")
        ax.set_title(wl(wl_), fontweight="bold", fontsize=10)
        ax.set_xlabel("k",fontsize=9); ax.set_ylabel("tok/s",fontsize=9)
        ax.tick_params(labelsize=8); ax.yaxis.grid(True, alpha=0.4)

    for ax in axes[len(wls):]: ax.axis("off")
    axes[len(wls)].legend(handles=handles_legend(METHOD_ORDER),
                           loc="center", fontsize=12, title="Method")
    fig.suptitle("Fig 9: Throughput Scaling — All Workloads × All Methods",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, out/"fig09_throughput_scaling_all.png")


def fig10_entropy_by_method(all_df, out):
    """Entropy per method — confirms method invariance."""
    all_df = all_df.copy()
    wls    = workloads_in(all_df)
    piv = (all_df.groupby(["workload","method"])["mean_entropy"].mean()
             .unstack().reindex(columns=[m for m in METHOD_ORDER
                                          if m in all_df["method"].unique()]))
    piv.index   = piv.index.map(wl)
    piv.columns = [ml(m) for m in piv.columns]

    fig, ax = plt.subplots(figsize=(13, 5))
    piv.plot(kind="bar", ax=ax,
             color=[COLORS[m] for m in METHOD_ORDER if ml(m) in piv.columns],
             width=0.7, rot=20)
    ax.set_ylabel("Mean Token Entropy")
    ax.set_title("Fig 10: Token Entropy — All 4 Methods × All Workloads\n"
                 "(Near-identical bars = entropy is a workload property, not method-dependent)",
                 fontweight="bold")
    ax.legend(title="Method"); ax.yaxis.grid(True, alpha=0.4)
    save(fig, out/"fig10_entropy_by_method.png")


# ═══════════════════════════════════════════════════════════════
# PHASE 2
# ═══════════════════════════════════════════════════════════════

def fig11_temperature_sweep(temp_df, out):
    """Temperature sweep — all 4 methods including draft_sd (Exp E)."""
    if temp_df is None or temp_df.empty: return
    wls   = workloads_in(temp_df)
    meths = methods_in(temp_df)   # will be ar/eagle3/ngram_sd per the experiment config

    for ycol, ylabel, stem in [
        ("tps_mean",         "Tokens / Second", "tps"),
        ("accept_rate_mean", "Acceptance Rate",  "acceptance"),
    ]:
        if ycol not in temp_df.columns: continue
        fig, axes = plt.subplots(1, len(meths), figsize=(6*len(meths), 5))
        if len(meths) == 1: axes = [axes]
        pal = sns.color_palette("tab10", len(wls))
        for ax, m in zip(axes, meths):
            sub = temp_df[temp_df["method"]==m]
            for j, wl_ in enumerate(wls):
                s = sub[sub["workload"]==wl_].groupby("temperature")[ycol].mean()
                if s.empty: continue
                ax.plot(s.index, s.values, marker="o", color=pal[j],
                        label=wl(wl_), linewidth=2)
            ax.set_title(ml(m), fontweight="bold")
            ax.set_xlabel("Temperature"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, ncol=2); ax.yaxis.grid(True, alpha=0.4)
        note = ""   # draft_sd now included in experiment E
        fig.suptitle(f"Fig 11: Temperature Sweep — {ylabel}\n{note}",
                     fontweight="bold")
        save(fig, out/f"fig11_temperature_{stem}.png")


def fig12_context_growth(ctx_df, out):
    """Context length scaling — all 4 methods including draft_sd (Exp F)."""
    if ctx_df is None or ctx_df.empty: return
    wls   = workloads_in(ctx_df)
    ncols = 4; nrows = math.ceil(len(wls)/ncols)

    for ycol, ylabel in [
        ("tps_mean",     "Tokens / Second"),
        ("speedup_vs_ar","Speedup vs AR"),
    ]:
        if ycol not in ctx_df.columns: continue
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*4))
        axes = axes.flatten()
        for idx, wl_ in enumerate(wls):
            ax = axes[idx]
            sub = ctx_df[ctx_df["workload"]==wl_]
            for m in methods_in(sub):
                s = sub[sub["method"]==m].groupby("max_prompt_tokens")[ycol].mean()
                ax.plot(s.index, s.values, marker="o", color=COLORS[m],
                        label=ml(m), linewidth=2)
            ax.set_title(wl(wl_), fontweight="bold", fontsize=10)
            ax.set_xlabel("Prompt tokens"); ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4)
        for ax in axes[len(wls):]: ax.axis("off")
        note = "(AR / Ngram-SD / Eagle3 only — Draft-SD excluded from experiment F)"
        fig.suptitle(f"Fig 12: Context Growth — {ylabel}\n{note}",
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        save(fig, out/f"fig12_context_{ycol}.png")


def fig13_draft_model_sweep(draft_df, main_agg, out):
    """
    Draft model family comparison.
    Adds AR baseline from main_agg since D_draft only contains draft_sd rows.
    Computes speedup_vs_ar by merging AR TPS.
    """
    if draft_df is None or draft_df.empty: return
    if "draft_model" not in draft_df.columns: return

    # Add AR baseline TPS for speedup computation
    if main_agg is not None and "tps_mean" in main_agg.columns:
        ar_base = (main_agg[main_agg["method"]=="ar"]
                   .groupby("workload")["tps_mean"].mean()
                   .reset_index().rename(columns={"tps_mean":"ar_tps"}))
        draft_df = draft_df.merge(ar_base, on="workload", how="left")
        draft_df["speedup_vs_ar"] = draft_df["tps_mean"] / draft_df["ar_tps"]

    models    = sorted(draft_df["draft_model"].dropna().unique())
    workloads = workloads_in(draft_df)
    pal       = dict(zip(models, sns.color_palette("Set2", max(len(models),3))))
    ks        = sorted(draft_df["k"].dropna().unique())

    # Show note if only one model ran
    model_note = (f"({len(models)} draft model(s) ran: {', '.join(m.split('/')[-1] for m in models)})"
                  if len(models) < 3 else "(Qwen3 0.6B / 1.7B / 4B)")

    for ycol, ylabel in [
        ("tps_mean",         "Tokens / Second"),
        ("accept_rate_mean", "Acceptance Rate"),
        ("speedup_vs_ar",    "Speedup vs AR"),
    ]:
        if ycol not in draft_df.columns: continue
        ncols = min(3, len(workloads))
        nrows = math.ceil(len(workloads)/max(ncols,1))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        axes = axes.flatten() if nrows*ncols > 1 else [axes]

        for idx, wl_ in enumerate(workloads):
            ax = axes[idx]
            sub = draft_df[draft_df["workload"]==wl_]
            for dm, ddf in sub.groupby("draft_model"):
                s = ddf.sort_values("k")
                ax.plot(s["k"], s[ycol], marker="o",
                        color=pal.get(dm,"#999"),
                        label=str(dm).split("/")[-1], linewidth=2)
            # Add AR reference line
            if main_agg is not None and ycol in ["tps_mean","speedup_vs_ar"]:
                ar_val = (main_agg[(main_agg["method"]=="ar")&
                                   (main_agg["workload"]==wl_)]["tps_mean"].mean())
                if ycol == "tps_mean" and not np.isnan(ar_val):
                    ax.axhline(ar_val, color=COLORS["ar"], linestyle="--",
                               linewidth=1.5, label="AR baseline")
                elif ycol == "speedup_vs_ar":
                    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
            ax.set_title(wl(wl_), fontweight="bold")
            ax.set_xticks(ks); ax.set_xlabel("k"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4)

        for ax in axes[len(workloads):]: ax.axis("off")
        fig.suptitle(f"Fig 13: Qwen3 Draft Model Family — {ylabel}\n{model_note}",
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        save(fig, out/f"fig13_draft_model_{ycol}.png")


def fig14_ngram_config(ngram_df, out):
    """Ngram lookup window sensitivity."""
    if ngram_df is None or ngram_df.empty: return
    if not {"ngram_lookup_min","ngram_lookup_max"}.issubset(ngram_df.columns): return
    wls = workloads_in(ngram_df)
    ngram_df = ngram_df.copy()
    ngram_df["config"] = ngram_df.apply(
        lambda r: f"k={int(r['k'])} win[{int(r['ngram_lookup_min'])}-{int(r['ngram_lookup_max'])}]",
        axis=1)
    cfgs = sorted(ngram_df["config"].unique())
    pal  = sns.color_palette("tab10", len(cfgs))

    for ycol, ylabel in [("tps_mean","Tokens/sec"),("accept_rate_mean","Acceptance Rate")]:
        if ycol not in ngram_df.columns: continue
        ncols = min(4, len(wls)); nrows = math.ceil(len(wls)/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = axes.flatten() if nrows*ncols>1 else [axes]
        for idx, wl_ in enumerate(wls):
            ax = axes[idx]
            sub = ngram_df[ngram_df["workload"]==wl_]
            for j, cfg in enumerate(cfgs):
                val = sub[sub["config"]==cfg][ycol].mean()
                ax.bar(j, val, color=pal[j], label=cfg, alpha=0.85)
            ax.set_title(wl(wl_), fontweight="bold")
            ax.set_xticks(range(len(cfgs)))
            ax.set_xticklabels(cfgs, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel); ax.yaxis.grid(True, alpha=0.4)
        for ax in axes[len(wls):]: ax.axis("off")
        fig.suptitle(f"Fig 14: Ngram Window Config Sensitivity — {ylabel}",
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        save(fig, out/f"fig14_ngram_config_{ycol}.png")


def fig15_multi_turn(turn_df, out):
    """Multi-turn depth effect — all 4 methods including draft_sd (Exp G)."""
    if turn_df is None or turn_df.empty: return
    if "num_turns" not in turn_df.columns: return
    wls   = workloads_in(turn_df)
    meths = methods_in(turn_df)

    for ycol, ylabel in [
        ("tps_mean","Tokens / Second"),
        ("accept_rate_mean","Acceptance Rate"),
        ("lat_mean","Mean Latency (s)"),
    ]:
        if ycol not in turn_df.columns: continue
        ncols = min(3, len(wls)); nrows = math.ceil(len(wls)/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        axes = axes.flatten() if nrows*ncols>1 else [axes]
        for idx, wl_ in enumerate(wls):
            ax = axes[idx]
            sub = turn_df[turn_df["workload"]==wl_]
            for m in meths:
                s = sub[sub["method"]==m].groupby("num_turns")[ycol].mean()
                if s.empty: continue
                ax.plot(s.index, s.values, marker="o", color=COLORS[m],
                        label=ml(m), linewidth=2)
            ax.set_title(wl(wl_), fontweight="bold")
            ax.set_xlabel("Number of Turns"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4)
        for ax in axes[len(wls):]: ax.axis("off")
        note = "(AR / Ngram-SD / Eagle3 — Draft-SD excluded from experiment G)"
        fig.suptitle(f"Fig 15: Multi-Turn Depth — {ylabel}\n{note}",
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        save(fig, out/f"fig15_multi_turn_{ycol}.png")


def fig16_positional_acceptance(pos_df, out):
    """Per-position acceptance decay — all 3 speculative methods."""
    if pos_df is None or pos_df.empty: return
    wls   = workloads_in(pos_df)
    meths = [m for m in ["eagle3","draft_sd","ngram_sd"]
             if m in pos_df["method"].unique()]
    pal   = sns.color_palette("tab10", len(wls))

    fig, axes = plt.subplots(1, len(meths), figsize=(6*len(meths), 5))
    if len(meths) == 1: axes = [axes]
    for ax, m in zip(axes, meths):
        sub = pos_df[(pos_df["method"]==m) & (pos_df["k"]==4)]
        for j, wl_ in enumerate(wls):
            s = (sub[sub["workload"]==wl_]
                 .groupby("position")["accept_rate_at_position"].mean())
            if s.empty: continue
            ax.plot(s.index, s.values, marker="o", color=pal[j],
                    label=wl(wl_), linewidth=2)
        ax.set_title(f"{ml(m)} — Positional Decay (k=4)", fontweight="bold")
        ax.set_xlabel("Draft Position (0=first token)")
        ax.set_ylabel("Acceptance Rate at Position")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="50% threshold")
        ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 16: Per-Position Acceptance Decay — All Speculative Methods",
                 fontweight="bold")
    save(fig, out/"fig16_positional_acceptance.png")


def fig17_acceptance_vs_entropy(all_df, out):
    """Acceptance rate vs entropy — all speculative methods."""
    spec = all_df[all_df["method"].isin(["eagle3","ngram_sd","draft_sd"])].copy()
    spec = spec.dropna(subset=["acceptance_rate","mean_entropy"])
    if spec.empty:
        print("  [skip] fig17: no acceptance_rate data (NaN for all rows)")
        return
    fig, ax = plt.subplots(figsize=(10,6))
    for m, sub in spec.groupby("method"):
        ax.scatter(sub["mean_entropy"], sub["acceptance_rate"],
                   color=COLORS[m], label=ml(m), alpha=0.35, s=15)
        x_v = sub["mean_entropy"].values; y_v = sub["acceptance_rate"].values
        if len(x_v) > 10:
            z  = np.polyfit(x_v, y_v, 1)
            xl = np.linspace(x_v.min(), x_v.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), color=COLORS[m], linewidth=2.5)
    ax.set_xlabel("Mean Token Entropy"); ax.set_ylabel("Acceptance Rate")
    ax.set_title("Fig 17: Acceptance Rate vs Token Entropy — All Phase 2 Runs",
                 fontweight="bold")
    ax.legend(title="Method"); ax.yaxis.grid(True, alpha=0.4)
    save(fig, out/"fig17_acceptance_vs_entropy.png")


def fig18_temperature_heatmap(temp_df, out):
    """Temperature × workload heatmap for acceptance rate."""
    if temp_df is None or temp_df.empty: return
    if "accept_rate_mean" not in temp_df.columns: return
    for m in [m for m in ["eagle3","ngram_sd","draft_sd"] if m in temp_df["method"].unique()]:
        sub = temp_df[(temp_df["method"]==m)&(temp_df["k"]==4)]
        if sub.empty: continue
        piv = sub.pivot_table(index="workload", columns="temperature",
                              values="accept_rate_mean", aggfunc="mean")
        piv.index = piv.index.map(wl)
        if piv.empty: continue
        fig, ax = plt.subplots(figsize=(9,5))
        sns.heatmap(piv.round(3), annot=True, fmt=".3f", cmap="RdYlGn",
                    ax=ax, linewidths=0.5, cbar_kws={"label":"Acceptance Rate"})
        ax.set_title(f"Fig 18: {ml(m)} — Acceptance Rate × Temperature × Workload",
                     fontweight="bold")
        ax.set_xlabel("Temperature"); ax.set_ylabel("")
        save(fig, out/f"fig18_temperature_heatmap_{m}.png")


def fig19_context_speedup_heatmap(ctx_df, out):
    """Context length × method speedup heatmap."""
    if ctx_df is None or ctx_df.empty: return
    if "speedup_vs_ar" not in ctx_df.columns: return
    for m in [m for m in ["eagle3","ngram_sd","draft_sd"] if m in ctx_df["method"].unique()]:
        sub = ctx_df[(ctx_df["method"]==m)&(ctx_df["k"]==4)]
        if sub.empty: continue
        piv = sub.pivot_table(index="workload", columns="max_prompt_tokens",
                              values="speedup_vs_ar", aggfunc="mean")
        piv.index = piv.index.map(wl)
        if piv.empty: continue
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(piv.round(2), annot=True, fmt=".2f", cmap="RdYlGn", center=1.0,
                    ax=ax, linewidths=0.5, cbar_kws={"label":"Speedup vs AR"})
        ax.set_title(f"Fig 19: {ml(m)} Speedup vs Context Length",
                     fontweight="bold")
        ax.set_xlabel("Prompt Tokens"); ax.set_ylabel("")
        save(fig, out/f"fig19_context_speedup_{m}.png")


def fig20_multi_turn_entropy(turn_df, out):
    """Multi-turn: how entropy evolves with turn depth."""
    if turn_df is None or turn_df.empty: return
    if "num_turns" not in turn_df.columns or "entropy_mean" not in turn_df.columns:
        return
    wls  = workloads_in(turn_df)
    meths = methods_in(turn_df)
    pal  = sns.color_palette("tab10", len(wls))

    fig, axes = plt.subplots(1, len(meths), figsize=(6*len(meths), 5))
    if len(meths) == 1: axes = [axes]
    for ax, m in zip(axes, meths):
        sub = turn_df[turn_df["method"]==m]
        for j, wl_ in enumerate(wls):
            s = sub[sub["workload"]==wl_].groupby("num_turns")["entropy_mean"].mean()
            if s.empty: continue
            ax.plot(s.index, s.values, marker="o", color=pal[j],
                    label=wl(wl_), linewidth=2)
        ax.set_title(f"{ml(m)} — Entropy vs Turn Depth", fontweight="bold")
        ax.set_xlabel("Number of Turns"); ax.set_ylabel("Mean Token Entropy")
        ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 20: Token Entropy Evolution with Turn Depth",
                 fontweight="bold")
    save(fig, out/"fig20_multi_turn_entropy.png")



# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS H / I / J / K  (figs 21-29)
# ═══════════════════════════════════════════════════════════════

def fig21_oracle_k(oracle_df, out):
    """Fig 21: Oracle k study — retrospective optimal k vs fixed k=4 (Exp H)."""
    if oracle_df is None or oracle_df.empty: return
    if not {"oracle_k_mean","oracle_gain_mean"}.issubset(oracle_df.columns): return
    wls  = workloads_in(oracle_df)
    spec = [m for m in ["eagle3","ngram_sd"] if m in oracle_df["method"].unique()]
    x, wb = np.arange(len(wls)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, m in enumerate(spec):
        sub  = oracle_df[oracle_df["method"]==m]
        vals = [sub[sub["workload"]==wl_]["oracle_k_mean"].mean() for wl_ in wls]
        axes[0].bar(x+(i-0.5)*wb, vals, width=wb*0.9, color=COLORS[m], label=ml(m), alpha=0.85)
    axes[0].axhline(4, color="gray", linestyle="--", linewidth=1.5, label="Fixed k=4")
    axes[0].set_xticks(x); axes[0].set_xticklabels([wl(w) for w in wls], rotation=20, ha="right")
    axes[0].set_ylabel("Oracle-Optimal k")
    axes[0].set_title("Retrospective Oracle k per Workload", fontweight="bold")
    axes[0].legend(); axes[0].yaxis.grid(True, alpha=0.4)
    for i, m in enumerate(spec):
        sub  = oracle_df[oracle_df["method"]==m]
        vals = [sub[sub["workload"]==wl_]["oracle_gain_mean"].mean() for wl_ in wls]
        axes[1].bar(x+(i-0.5)*wb, vals, width=wb*0.9, color=COLORS[m], label=ml(m), alpha=0.85)
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels([wl(w) for w in wls], rotation=20, ha="right")
    axes[1].set_ylabel("Oracle Gain vs Fixed k=4 (tokens/draft)")
    axes[1].set_title("Adaptive Controller Potential Gain", fontweight="bold")
    axes[1].legend(); axes[1].yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 21: Oracle k Study (Exp H)", fontweight="bold")
    save(fig, out/"fig21_oracle_k_analysis.png")


def fig22_oracle_k_by_bucket(oracle_bucket_df, out):
    """Fig 22: Oracle gain by entropy bucket — low entropy benefits most (Exp H)."""
    if oracle_bucket_df is None or oracle_bucket_df.empty: return
    if "entropy_bucket" not in oracle_bucket_df.columns: return
    if "oracle_gain_mean" not in oracle_bucket_df.columns: return
    buckets = ["low","medium","high"]
    spec    = [m for m in ["eagle3","ngram_sd"] if m in oracle_bucket_df["method"].unique()]
    pal_b   = {"low":"#2196F3","medium":"#FF9800","high":"#F44336"}
    fig, axes = plt.subplots(1, len(spec), figsize=(6*len(spec), 5))
    if len(spec)==1: axes=[axes]
    for ax, m in zip(axes, spec):
        sub = oracle_bucket_df[oracle_bucket_df["method"]==m]
        wls = workloads_in(sub); x=np.arange(len(wls)); wb=0.25
        for i, bucket in enumerate(buckets):
            bsub = sub[sub["entropy_bucket"]==bucket]
            vals = [bsub[bsub["workload"]==wl_]["oracle_gain_mean"].mean() for wl_ in wls]
            ax.bar(x+(i-1)*wb, vals, width=wb*0.9, color=pal_b[bucket], label=f"{bucket} entropy", alpha=0.85)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels([wl(w) for w in wls], rotation=20, ha="right")
        ax.set_ylabel("Oracle Gain vs k=4")
        ax.set_title(f"{ml(m)} Oracle Gain by Entropy Bucket", fontweight="bold")
        ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 22: Oracle Gain by Entropy Bucket (Exp H)", fontweight="bold")
    save(fig, out/"fig22_oracle_k_by_bucket.png")


def fig23_entropy_bucket_speedup(bucket_df, out):
    """Fig 23: Speedup within each entropy bucket — the adaptive controller lookup table (Exp I)."""
    if bucket_df is None or bucket_df.empty: return
    if not {"entropy_bucket","speedup_vs_ar"}.issubset(bucket_df.columns): return
    buckets = ["low","medium","high"]
    pal_b   = {"low":"#2196F3","medium":"#FF9800","high":"#F44336"}
    spec    = [m for m in ["eagle3","ngram_sd","draft_sd"] if m in bucket_df["method"].unique()]
    ks      = sorted(bucket_df["k"].dropna().unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, bucket in zip(axes, buckets):
        bsub = bucket_df[bucket_df["entropy_bucket"]==bucket]
        for m in spec:
            msub = bsub[bsub["method"]==m].groupby("k")["speedup_vs_ar"].mean()
            if msub.empty: continue
            ax.plot(msub.index, msub.values, marker="o", color=COLORS[m], label=ml(m), linewidth=2.5)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{bucket.capitalize()} Entropy Bucket", fontweight="bold", color=pal_b[bucket])
        ax.set_xlabel("k"); ax.set_ylabel("Speedup vs AR")
        ax.set_xticks(ks); ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 23: Speedup by Entropy Bucket (Exp I) — Adaptive Controller Lookup Table", fontweight="bold")
    save(fig, out/"fig23_entropy_bucket_speedup.png")


def fig24_entropy_bucket_heatmap(bucket_df, out):
    """Fig 24: Best speedup heatmap — entropy bucket x workload (Exp I)."""
    if bucket_df is None or bucket_df.empty: return
    if not {"entropy_bucket","speedup_vs_ar"}.issubset(bucket_df.columns): return
    for m in [m for m in ["eagle3","ngram_sd"] if m in bucket_df["method"].unique()]:
        sub = bucket_df[bucket_df["method"]==m]
        piv = sub.groupby(["workload","entropy_bucket"])["speedup_vs_ar"].max().unstack()
        for b in ["low","medium","high"]:
            if b not in piv.columns: piv[b] = float("nan")
        piv = piv[["low","medium","high"]]
        piv.index = piv.index.map(wl)
        if piv.empty: continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(piv.round(2), annot=True, fmt=".2f", cmap="RdYlGn", center=1.0,
                    ax=ax, linewidths=0.5, cbar_kws={"label":"Best Speedup vs AR"})
        ax.set_title(f"Fig 24: {ml(m)} Peak Speedup by Entropy Bucket x Workload (Exp I)", fontweight="bold")
        ax.set_xlabel("Entropy Bucket"); ax.set_ylabel("")
        save(fig, out/f"fig24_entropy_bucket_heatmap_{m}.png")


def fig25_intra_sequence_windows(intra_df, out):
    """Fig 25: Intra-sequence entropy window analysis — how entropy evolves within a generation (Exp J)."""
    if intra_df is None or intra_df.empty: return
    win_cols = [c for c in intra_df.columns if c.startswith("entropy_w") and c.endswith("_mean")]
    if not win_cols: return
    wls    = workloads_in(intra_df)
    ar_sub = intra_df[intra_df["method"]=="ar"]
    if ar_sub.empty: return
    fig, ax = plt.subplots(figsize=(13, 5))
    x, wb = np.arange(len(wls)), 0.25
    colors_w = ["#5C6BC0","#26A69A","#EF5350"]
    labels_w = ["Window 0 (first third)","Window 1 (middle)","Window 2 (last third)"]
    for i, (col, label, c) in enumerate(zip(win_cols[:3], labels_w, colors_w)):
        vals = [ar_sub[ar_sub["workload"]==wl_][col].mean() for wl_ in wls]
        ax.bar(x+(i-1)*wb, vals, width=wb*0.9, color=c, label=label, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([wl(w) for w in wls], rotation=20, ha="right")
    ax.set_ylabel("Mean Token Entropy")
    ax.set_title("Fig 25: Intra-Sequence Entropy Transitions (AR baseline, Exp J)", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.4)
    save(fig, out/"fig25_intra_sequence_windows.png")


def fig26_entropy_trend(intra_df, out):
    """Fig 26: Entropy trend (slope) by workload — negative means settling (Exp J)."""
    if intra_df is None or intra_df.empty: return
    col = next((c for c in ["entropy_trend_mean","entropy_trend"] if c in intra_df.columns), None)
    if col is None: return
    wls  = workloads_in(intra_df)
    meths = methods_in(intra_df)
    x, wb = np.arange(len(wls)), 0.2
    n = len(meths); offs = np.linspace(-(n-1)/2,(n-1)/2,n)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(meths):
        sub  = intra_df[intra_df["method"]==m]
        vals = [sub[sub["workload"]==wl_][col].mean() for wl_ in wls]
        ax.bar(x+offs[i]*wb, vals, width=wb*0.9, color=COLORS.get(m,"#999"), label=ml(m), alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels([wl(w) for w in wls], rotation=20, ha="right")
    ax.set_ylabel("Entropy Trend (nats/token) — negative=settling, positive=diverging")
    ax.set_title("Fig 26: Entropy Trend Within Generation (Exp J)", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.4)
    save(fig, out/"fig26_entropy_trend.png")


def fig27_rejection_heatmap(rej_df, out):
    """Fig 27: Rejection metrics heatmap — workload x method at k=4 (Exp K)."""
    if rej_df is None or rej_df.empty: return
    rej_metrics = {
        "rejection_rate_mean":      "Rejection Rate",
        "rejection_severity_mean":  "Rejection Severity (pos-0 rate)",
        "first_rejection_pos_mean": "First Rejection Position",
        "decay_slope_mean":         "Acceptance Decay Slope",
        "rejection_conc_mean":      "Rejection Concentration",
    }
    for col, title in rej_metrics.items():
        if col not in rej_df.columns: continue
        sub = rej_df[(rej_df["method"].isin(["eagle3","ngram_sd","draft_sd"])) & (rej_df["k"]==4)]
        if sub.empty: continue
        piv = sub.pivot_table(index="workload", columns="method", values=col, aggfunc="mean")
        piv.index = piv.index.map(wl); piv.columns = [ml(m) for m in piv.columns]
        if piv.empty: continue
        cmap = "RdYlGn_r" if "rate" in col or "severity" in col else "RdYlGn"
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(piv.round(3), annot=True, fmt=".3f", cmap=cmap,
                    ax=ax, linewidths=0.5, cbar_kws={"label":title,"shrink":0.85})
        ax.set_title(f"Fig 27: {title} — Workload x Method (k=4, Exp K)", fontweight="bold")
        ax.set_xlabel("Method"); ax.set_ylabel("")
        fname = col.replace("_mean","")
        save(fig, out/f"fig27_rejection_{fname}.png")


def fig28_rejection_vs_k(rej_df, out):
    """Fig 28: Rejection metrics vs k — how does increasing k affect rejection patterns? (Exp K)."""
    if rej_df is None or rej_df.empty: return
    spec = [m for m in ["eagle3","ngram_sd","draft_sd"] if m in rej_df["method"].unique()]
    for ycol, ylabel in [
        ("rejection_rate_mean",     "Rejection Rate"),
        ("decay_slope_mean",        "Acceptance Decay Slope"),
        ("first_rejection_pos_mean","First Rejection Position"),
    ]:
        if ycol not in rej_df.columns: continue
        ks  = sorted(rej_df["k"].dropna().unique())
        fig, ax = plt.subplots(figsize=(9, 5))
        for m in spec:
            sub = rej_df[(rej_df["method"]==m)].groupby("k")[ycol].mean()
            if sub.empty: continue
            ax.plot(sub.index, sub.values, marker="o", color=COLORS[m], label=ml(m), linewidth=2.5)
        ax.set_xlabel("Speculation Depth k"); ax.set_ylabel(ylabel)
        ax.set_title(f"Fig 28: {ylabel} vs k (Exp K)", fontweight="bold")
        ax.set_xticks(ks); ax.legend(); ax.yaxis.grid(True, alpha=0.4)
        fname = ycol.replace("_mean","")
        save(fig, out/f"fig28_rejection_vs_k_{fname}.png")


def fig29_rejection_temperature(rej_df, out):
    """Fig 29: Temperature effect on rejection patterns — T=0.0 vs T=0.6 (Exp K)."""
    if rej_df is None or rej_df.empty: return
    if "temperature" not in rej_df.columns: return
    if len(rej_df["temperature"].unique()) < 2: return
    spec    = [m for m in ["eagle3","ngram_sd"] if m in rej_df["method"].unique()]
    ycols   = [c for c in ["rejection_rate_mean","rejection_severity_mean","decay_slope_mean"]
               if c in rej_df.columns]
    if not ycols: return
    wls = workloads_in(rej_df)
    temp_colors = {0.0:"#2196F3", 0.6:"#F44336"}
    fig, axes = plt.subplots(1, len(ycols), figsize=(6*len(ycols), 5))
    if len(ycols)==1: axes=[axes]
    for ax, ycol in zip(axes, ycols):
        x, wb = np.arange(len(wls)), 0.2; idx = 0
        for m in spec:
            for t in sorted(rej_df["temperature"].unique()):
                sub  = rej_df[(rej_df["method"]==m)&(rej_df["temperature"]==t)&(rej_df["k"]==4)]
                vals = [sub[sub["workload"]==wl_][ycol].mean() for wl_ in wls]
                ax.bar(x+(idx-len(spec)*2/2+0.5)*wb, vals, width=wb*0.9,
                       color=temp_colors.get(t,"#999"), label=f"{ml(m)} T={t}", alpha=0.85)
                idx += 1
        ax.set_xticks(x); ax.set_xticklabels([wl(w) for w in wls], rotation=20, ha="right")
        ax.set_ylabel(ycol.replace("_mean","").replace("_"," ").title())
        ax.set_title("T=0.0 vs T=0.6", fontweight="bold")
        ax.legend(fontsize=7, ncol=2); ax.yaxis.grid(True, alpha=0.4)
    fig.suptitle("Fig 29: Rejection Metrics Temperature Effect (Exp K)", fontweight="bold")
    save(fig, out/"fig29_rejection_temperature.png")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir",     default="results/plots")
    args = ap.parse_args()

    rd  = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    agg      = rp(rd, "aggregate_by_workload_method.csv")
    all_df   = rp(rd, "all_runs.csv")
    temp_df  = rp(rd, "aggregate_temperature_sweep.csv")
    ctx_df   = rp(rd, "aggregate_context_growth.csv")
    draft_df = rp(rd, "aggregate_draft_model_sweep.csv")
    ngram_df = rp(rd, "aggregate_ngram_config_sweep.csv")
    turn_df  = rp(rd, "aggregate_multi_turn.csv")
    pos_df   = rp(rd, "aggregate_positional_acceptance.csv")

    if agg is None:
        raise SystemExit(f"aggregate_by_workload_method.csv not found in {rd}")

    print("Phase 1 plots...")
    fig01_throughput_bar(agg, out)
    fig02_speedup_heatmap(agg, out)
    fig03_k_sensitivity(agg, out)
    if all_df is not None:
        fig04_latency_cdf(all_df, out)
        fig05_structural_heatmaps(all_df, out)
        fig10_entropy_by_method(all_df, out)
        fig17_acceptance_vs_entropy(all_df, out)
    fig06_entropy_speedup_scatter(agg, out)
    fig07_draft_sd_analysis(agg, out)
    fig08_ngram_beats_eagle3(agg, out)
    fig09_throughput_scaling(agg, out)

    print("Phase 2 plots...")
    fig11_temperature_sweep(temp_df, out)
    fig12_context_growth(ctx_df, out)
    fig13_draft_model_sweep(draft_df, agg, out)   # passes main_agg for AR baseline
    fig14_ngram_config(ngram_df, out)
    fig15_multi_turn(turn_df, out)
    fig16_positional_acceptance(pos_df, out)
    fig18_temperature_heatmap(temp_df, out)
    fig19_context_speedup_heatmap(ctx_df, out)
    fig20_multi_turn_entropy(turn_df, out)

    # Load new experiment CSVs
    oracle_df        = rp(rd, "aggregate_oracle_k.csv")
    oracle_bucket_df = rp(rd, "aggregate_oracle_k_by_bucket.csv")
    bucket_df        = rp(rd, "aggregate_entropy_bucket.csv")
    intra_df         = rp(rd, "aggregate_intra_sequence.csv")
    rej_df           = rp(rd, "aggregate_rejection_study.csv")

    print("Experiments H / I / J / K plots...")
    fig21_oracle_k(oracle_df, out)
    fig22_oracle_k_by_bucket(oracle_bucket_df, out)
    fig23_entropy_bucket_speedup(bucket_df, out)
    fig24_entropy_bucket_heatmap(bucket_df, out)
    fig25_intra_sequence_windows(intra_df, out)
    fig26_entropy_trend(intra_df, out)
    fig27_rejection_heatmap(rej_df, out)
    fig28_rejection_vs_k(rej_df, out)
    fig29_rejection_temperature(rej_df, out)

    n = len(list(out.rglob("*.png")))
    print(f"\nDone — {n} plots in {out}")


if __name__ == "__main__":
    main()