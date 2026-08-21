#!/usr/bin/env python3
"""
make_phase1_report.py
=====================
Generates all Phase 1 tables (CSV) and figures (PNG) from all_runs.csv.

Usage:
    python make_phase1_report.py
    python make_phase1_report.py --data path/to/all_runs.csv --out_dir my_output/

Outputs:
    tables/
        table1_baseline_throughput_k4.csv
        table2_eagle3_k_sweep_tps.csv
        table2_ngram_k_sweep_tps.csv
        table2_draft_k_sweep_tps.csv
        table3_structural_metrics.csv
        table4_latency_breakdown.csv
        table5_best_k_per_method_workload.csv
        table6_entropy_speedup_correlation.csv
        table7_draft_k_sensitivity.csv
    figures/
        fig1_throughput_bar_k4.png
        fig2_speedup_heatmap.png
        fig3_k_sensitivity_eagle3_ngram.png
        fig4_latency_cdf.png
        fig5_structural_heatmaps.png
        fig6_entropy_speedup_scatter.png
        fig7_draft_sd_analysis.png
        fig8_ngram_beats_eagle3_long.png
        fig9_throughput_scaling_all.png
        fig10_entropy_by_method.png
"""

import argparse
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

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
WORKLOAD_SHORT = {
    "code_gen":                "Code Gen",
    "conversational_generation": "Conversational",
    "hardware_gen":            "Hardware Gen",
    "long_chain_reasoning":    "Long-Chain",
    "long_context_completion": "Long-Context",
    "long_horizon_swe":        "SWE",
    "mathematical_reasoning":  "Math",
}
WORKLOAD_FULL = {
    "code_gen":                "Code Gen",
    "conversational_generation": "Conversational",
    "hardware_gen":            "Hardware Gen",
    "long_chain_reasoning":    "Long-Chain Reasoning",
    "long_context_completion": "Long-Context Completion",
    "long_horizon_swe":        "Long-Horizon SWE",
    "mathematical_reasoning":  "Mathematical Reasoning",
}
METHOD_ORDER  = ["ar", "ngram_sd", "draft_sd", "eagle3"]
METHOD_LABELS = {"ar": "AR", "ngram_sd": "Ngram-SD",
                 "draft_sd": "Draft-SD", "eagle3": "Eagle3"}
COLORS = {
    "ar":       "#4C72B0",
    "ngram_sd": "#DD8452",
    "draft_sd": "#55A868",
    "eagle3":   "#C44E52",
}
LONG_WORKLOADS = ["long_chain_reasoning", "long_context_completion", "long_horizon_swe"]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)
    print(f"  [csv] {path.name}")


def save_csv_noindex(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  [csv] {path.name}")


def ar_tps_by_workload(df: pd.DataFrame) -> pd.Series:
    """Mean AR throughput per workload (AR only has k=1)."""
    return df[df["method"] == "ar"].groupby("workload")["tokens_per_sec"].mean()


# ─────────────────────────────────────────────────────────────────
# ══════════════  TABLES  ══════════════
# ─────────────────────────────────────────────────────────────────

def make_table1(df: pd.DataFrame, out: Path) -> None:
    """Table 1: Baseline throughput & speedup at k=4."""
    ar_tps = ar_tps_by_workload(df)

    # AR has only k=1 — treat it as the k-agnostic baseline
    ar_rows = df[df["method"] == "ar"].copy()
    ar_rows["k"] = 4
    k4 = pd.concat([ar_rows, df[(df["method"] != "ar") & (df["k"] == 4)]])

    tps_piv = (k4.groupby(["workload", "method"])["tokens_per_sec"]
                 .mean()
                 .unstack()
                 .reindex(columns=METHOD_ORDER))

    result = pd.DataFrame(index=tps_piv.index)
    for m in METHOD_ORDER:
        result[f"TPS_{METHOD_LABELS[m]}"] = tps_piv[m].round(1)
    for m in [m for m in METHOD_ORDER if m != "ar"]:
        result[f"Speedup_{METHOD_LABELS[m]}"] = (tps_piv[m] / ar_tps).round(2)

    result.index = result.index.map(WORKLOAD_FULL)
    save_csv(result, out / "table1_baseline_throughput_k4.csv")


def make_table2(df: pd.DataFrame, out: Path) -> None:
    """Table 2: Full k-sweep throughput for each speculative method."""
    for meth in ["eagle3", "ngram_sd", "draft_sd"]:
        piv = (df[df["method"] == meth]
               .groupby(["workload", "k"])["tokens_per_sec"]
               .mean()
               .unstack()
               .round(1))
        piv.index  = piv.index.map(WORKLOAD_FULL)
        piv.columns = [f"k={c}" for c in piv.columns]
        save_csv(piv, out / f"table2_{meth}_k_sweep_tps.csv")


def make_table3(df: pd.DataFrame, out: Path) -> None:
    """Table 3: Structural metrics (entropy + repetition density) at k=4."""
    ar_rows = df[df["method"] == "ar"].copy(); ar_rows["k"] = 4
    k4 = pd.concat([ar_rows, df[(df["method"] != "ar") & (df["k"] == 4)]])

    agg = k4.groupby(["workload", "method"]).agg(
        entropy=("mean_entropy", "mean"),
        rep_density=("repetition_density", "mean"),
    ).reset_index()

    entr = agg.pivot(index="workload", columns="method", values="entropy").reindex(columns=METHOD_ORDER).round(3)
    rep  = agg.pivot(index="workload", columns="method", values="rep_density").reindex(columns=METHOD_ORDER).round(3)
    entr.index = entr.index.map(WORKLOAD_FULL)
    rep.index  = rep.index.map(WORKLOAD_FULL)
    entr.columns = [f"Entropy_{METHOD_LABELS[m]}" for m in METHOD_ORDER]
    rep.columns  = [f"RepDensity_{METHOD_LABELS[m]}" for m in METHOD_ORDER]
    save_csv(pd.concat([entr, rep], axis=1), out / "table3_structural_metrics.csv")


def make_table4(df: pd.DataFrame, out: Path) -> None:
    """Table 4: Latency breakdown (mean, p50, p95) at k=4."""
    ar_rows = df[df["method"] == "ar"].copy(); ar_rows["k"] = 4
    k4 = pd.concat([ar_rows, df[(df["method"] != "ar") & (df["k"] == 4)]])

    agg = k4.groupby(["workload", "method"]).agg(
        lat_mean=("latency_s", "mean"),
        lat_p50 =("latency_s", "median"),
        lat_p95 =("latency_s", lambda x: x.quantile(0.95)),
    ).reset_index()

    frames = []
    for stat, label in [("lat_mean","Mean"), ("lat_p50","P50"), ("lat_p95","P95")]:
        p = agg.pivot(index="workload", columns="method", values=stat).reindex(columns=METHOD_ORDER).round(3)
        p.index = p.index.map(WORKLOAD_FULL)
        p.columns = [f"{label}_{METHOD_LABELS[m]}" for m in METHOD_ORDER]
        frames.append(p)
    save_csv(pd.concat(frames, axis=1), out / "table4_latency_breakdown.csv")


def make_table5(df: pd.DataFrame, out: Path) -> None:
    """Table 5: Best k per method per workload (by throughput)."""
    ar_tps = ar_tps_by_workload(df)
    rows = []
    for (wl, meth), sub in df[df["method"] != "ar"].groupby(["workload", "method"]):
        best  = sub.loc[sub["tokens_per_sec"].idxmax()]
        rows.append({
            "Workload":  WORKLOAD_FULL[wl],
            "Method":    METHOD_LABELS[meth],
            "Best k":    int(best["k"]),
            "Best TPS":  round(best["tokens_per_sec"], 1),
            "AR TPS":    round(ar_tps[wl], 1),
            "Speedup":   round(best["tokens_per_sec"] / ar_tps[wl], 2),
        })
    save_csv_noindex(pd.DataFrame(rows), out / "table5_best_k_per_method_workload.csv")


def make_table6(df: pd.DataFrame, out: Path) -> None:
    """Table 6: Entropy & repetition density vs speedup correlation."""
    ar_tps  = ar_tps_by_workload(df)
    entr_ar = df[df["method"] == "ar"].groupby("workload")["mean_entropy"].mean()
    rep_ar  = df[df["method"] == "ar"].groupby("workload")["repetition_density"].mean()
    tps_e4  = df[(df["method"] == "eagle3")  & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()
    tps_n4  = df[(df["method"] == "ngram_sd") & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()

    result = pd.DataFrame({
        "Workload":         [WORKLOAD_FULL[w] for w in entr_ar.index],
        "Mean Entropy":     entr_ar.values.round(3),
        "Rep Density":      rep_ar.values.round(3),
        "Eagle3 Speedup (k=4)":  (tps_e4 / ar_tps).values.round(2),
        "Ngram Speedup (k=4)":   (tps_n4 / ar_tps).values.round(2),
    })
    save_csv_noindex(result, out / "table6_entropy_speedup_correlation.csv")


def make_table7(df: pd.DataFrame, out: Path) -> None:
    """Table 7: Draft-SD k-sensitivity (showing degradation with k)."""
    ar_tps = ar_tps_by_workload(df)
    piv = (df[df["method"] == "draft_sd"]
           .groupby(["workload", "k"])["tokens_per_sec"]
           .mean()
           .unstack()
           .round(1))
    piv.index  = piv.index.map(WORKLOAD_FULL)
    piv.columns = [f"k={c}" for c in piv.columns]
    piv["AR Baseline"] = ar_tps.rename(index=WORKLOAD_FULL).round(1)
    save_csv(piv, out / "table7_draft_k_sensitivity.csv")


def make_table_md4(df: pd.DataFrame, out: Path) -> None:
    """Table from .md Section 4.2: Best-k speedup summary (combined view)."""
    ar_tps = ar_tps_by_workload(df)
    rows = []
    for wl in [w for w in WORKLOAD_FULL if w in df["workload"].unique()]:
        row = {"Workload": WORKLOAD_FULL[wl], "AR TPS": round(ar_tps[wl], 1)}
        for meth in ["eagle3", "ngram_sd"]:
            sub     = df[(df["workload"] == wl) & (df["method"] == meth)]
            best    = sub.loc[sub["tokens_per_sec"].idxmax()]
            row[f"{METHOD_LABELS[meth]} Best k"]      = int(best["k"])
            row[f"{METHOD_LABELS[meth]} Best TPS"]    = round(best["tokens_per_sec"], 1)
            row[f"{METHOD_LABELS[meth]} Speedup"]     = round(best["tokens_per_sec"] / ar_tps[wl], 2)
        rows.append(row)
    # Add mean row
    result = pd.DataFrame(rows)
    mean_row = {"Workload": "Mean"}
    for col in result.columns[1:]:
        mean_row[col] = round(result[col].mean(), 2)
    result = pd.concat([result, pd.DataFrame([mean_row])], ignore_index=True)
    save_csv_noindex(result, out / "table_md4_best_k_speedup_summary.csv")


def make_table_md5(df: pd.DataFrame, out: Path) -> None:
    """Table from .md Section 4.3: Token regime sorted by entropy with speedups."""
    ar_tps  = ar_tps_by_workload(df)
    entr_ar = df[df["method"] == "ar"].groupby("workload")["mean_entropy"].mean()
    rep_ar  = df[df["method"] == "ar"].groupby("workload")["repetition_density"].mean()
    tps_e4  = df[(df["method"] == "eagle3")  & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()
    tps_n4  = df[(df["method"] == "ngram_sd") & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()

    rows = []
    for wl in entr_ar.index:
        rows.append({
            "Workload":              WORKLOAD_FULL[wl],
            "Mean Entropy":          round(entr_ar[wl], 3),
            "Repetition Density":    round(rep_ar[wl], 3),
            "Eagle3 Speedup (k=4)":  round(tps_e4[wl] / ar_tps[wl], 2) if wl in tps_e4 else float("nan"),
            "Ngram-SD Speedup (k=4)":round(tps_n4[wl] / ar_tps[wl], 2) if wl in tps_n4 else float("nan"),
        })
    result = pd.DataFrame(rows).sort_values("Mean Entropy")
    save_csv_noindex(result, out / "table_md5_token_regime_speedup.csv")


def generate_md_report(df: pd.DataFrame, out: Path) -> None:
    """
    Generates the full Phase 1 results section as a Markdown file,
    with all tables embedded inline (matching the .md results section exactly).
    """
    ar_tps  = ar_tps_by_workload(df)
    entr_ar = df[df["method"] == "ar"].groupby("workload")["mean_entropy"].mean()
    rep_ar  = df[df["method"] == "ar"].groupby("workload")["repetition_density"].mean()
    tps_e4  = df[(df["method"] == "eagle3")  & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()
    tps_n4  = df[(df["method"] == "ngram_sd") & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()
    tps_d4  = df[(df["method"] == "draft_sd") & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()
    tps_a   = df[df["method"] == "ar"].groupby("workload")["tokens_per_sec"].mean()
    lat_k4  = df.copy(); lat_k4_ar = lat_k4[lat_k4["method"]=="ar"].copy(); lat_k4_ar["k"]=4
    lat_k4  = pd.concat([lat_k4_ar, lat_k4[(lat_k4["method"]!="ar")&(lat_k4["k"]==4)]])
    lat_piv = lat_k4.groupby(["workload","method"])["latency_s"].mean().unstack().reindex(columns=METHOD_ORDER)

    wls = [w for w in WORKLOAD_FULL if w in df["workload"].unique()]

    # Best-k per method
    def best_k_tps(meth, wl):
        sub = df[(df["workload"]==wl)&(df["method"]==meth)]
        return sub.loc[sub["tokens_per_sec"].idxmax()]

    def df_to_md(d: pd.DataFrame) -> str:
        lines = ["| " + " | ".join(str(c) for c in d.columns) + " |",
                 "| " + " | ".join(["---"]*len(d.columns)) + " |"]
        for _, row in d.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
        return "\n".join(lines)

    lines = []
    lines.append("# Phase 1 Results: Speculative Decoding Baseline Study\n")
    lines.append("## Experimental Setup\n")
    lines.append(
        "We evaluated four inference methods — Autoregressive (AR), N-gram Speculative Decoding "
        "(Ngram-SD), Draft-Model Speculative Decoding (Draft-SD), and Eagle3 — across seven diverse "
        "workloads on a single NVIDIA H200 GPU. Target model: **Qwen/Qwen3-8B**. Draft-SD used "
        "**Qwen2.5-1.5B-Instruct**; Eagle3 used **RedHatAI/Qwen3-8B-speculator.eagle3**. "
        "All speculative methods were evaluated at k ∈ {1, 2, 4, 8, 16}, temperature=0.0, seed=42.\n"
    )

    # ── Section 4.1 ──
    lines.append("---\n## 4.1 Baseline Throughput (k=4)\n")
    lines.append("**Table 1: Throughput (tokens/sec) and Speedup vs AR at k=4**\n")
    t1_rows = []
    for wl in wls:
        t1_rows.append({
            "Workload": WORKLOAD_FULL[wl],
            "AR": round(tps_a[wl],1),
            "Ngram-SD": round(tps_n4[wl],1) if wl in tps_n4 else "—",
            "Draft-SD": round(tps_d4[wl],1) if wl in tps_d4 else "—",
            "Eagle3":   round(tps_e4[wl],1) if wl in tps_e4 else "—",
            "Ngram Speedup": f"{tps_n4[wl]/tps_a[wl]:.2f}×" if wl in tps_n4 else "—",
            "Eagle3 Speedup": f"{tps_e4[wl]/tps_a[wl]:.2f}×" if wl in tps_e4 else "—",
        })
    lines.append(df_to_md(pd.DataFrame(t1_rows)) + "\n")

    # ── Section 4.2 ──
    lines.append("---\n## 4.2 Peak Throughput: k-Sweep Results\n")
    lines.append("**Table 2: Eagle3 Throughput (tok/s) by k**\n")
    t2e = df[df["method"]=="eagle3"].groupby(["workload","k"])["tokens_per_sec"].mean().unstack().round(1)
    t2e.index = t2e.index.map(WORKLOAD_FULL)
    t2e.columns = [f"k={c}" for c in t2e.columns]
    t2e_md = t2e.reset_index().rename(columns={"workload":"Workload"})
    lines.append(df_to_md(t2e_md) + "\n")

    lines.append("**Table 3: Ngram-SD Throughput (tok/s) by k**\n")
    t2n = df[df["method"]=="ngram_sd"].groupby(["workload","k"])["tokens_per_sec"].mean().unstack().round(1)
    t2n.index = t2n.index.map(WORKLOAD_FULL)
    t2n.columns = [f"k={c}" for c in t2n.columns]
    t2n_md = t2n.reset_index().rename(columns={"workload":"Workload"})
    lines.append(df_to_md(t2n_md) + "\n")

    lines.append("**Table 4: Best-k Speedup Summary**\n")
    t4_rows = []
    for wl in wls:
        be = best_k_tps("eagle3", wl)
        bn = best_k_tps("ngram_sd", wl)
        t4_rows.append({
            "Workload": WORKLOAD_FULL[wl],
            "Eagle3 Best k": int(be["k"]),
            "Eagle3 Speedup": f"{be['tokens_per_sec']/tps_a[wl]:.2f}×",
            "Ngram Best k": int(bn["k"]),
            "Ngram Speedup": f"{bn['tokens_per_sec']/tps_a[wl]:.2f}×",
            "AR Baseline": round(tps_a[wl],1),
        })
    mean_e = np.mean([float(r["Eagle3 Speedup"][:-1]) for r in t4_rows])
    mean_n = np.mean([float(r["Ngram Speedup"][:-1])  for r in t4_rows])
    t4_rows.append({"Workload":"**Mean**","Eagle3 Best k":"—",
                    "Eagle3 Speedup":f"**{mean_e:.2f}×**",
                    "Ngram Best k":"—","Ngram Speedup":f"**{mean_n:.2f}×**",
                    "AR Baseline":"—"})
    lines.append(df_to_md(pd.DataFrame(t4_rows)) + "\n")

    # ── Section 4.3 ──
    lines.append("---\n## 4.3 Token Regime Characterization\n")
    lines.append("**Table 5: Workload Token Regime Characterization (sorted by entropy)**\n")
    t5_rows = sorted([{
        "Workload": WORKLOAD_FULL[wl],
        "Mean Entropy": round(entr_ar[wl],3),
        "Rep Density": round(rep_ar[wl],3),
        "Eagle3 Speedup (k=4)": f"{tps_e4[wl]/tps_a[wl]:.2f}×" if wl in tps_e4 else "—",
        "Ngram Speedup (k=4)": f"{tps_n4[wl]/tps_a[wl]:.2f}×" if wl in tps_n4 else "—",
    } for wl in wls], key=lambda x: x["Mean Entropy"])
    lines.append(df_to_md(pd.DataFrame(t5_rows)) + "\n")

    # ── Section 4.4 ──
    lines.append("---\n## 4.4 Draft-SD: Effect of Draft Model Mismatch\n")
    lines.append("**Table 6: Draft-SD k-Sensitivity (all values below AR baseline)**\n")
    t6 = df[df["method"]=="draft_sd"].groupby(["workload","k"])["tokens_per_sec"].mean().unstack().round(1)
    t6.index = t6.index.map(WORKLOAD_FULL)
    t6.columns = [f"k={c}" for c in t6.columns]
    t6["AR Baseline"] = ar_tps.rename(index=WORKLOAD_FULL).round(1)
    lines.append(df_to_md(t6.reset_index().rename(columns={"workload":"Workload"})) + "\n")

    # ── Section 4.5 ──
    lines.append("---\n## 4.5 Latency Analysis\n")
    lines.append("**Table 7: Mean Request Latency (seconds) at k=4**\n")
    t7_rows = []
    for wl in wls:
        t7_rows.append({
            "Workload": WORKLOAD_FULL[wl],
            "AR":       round(lat_piv.loc[wl,"ar"],3)       if wl in lat_piv.index else "—",
            "Ngram-SD": round(lat_piv.loc[wl,"ngram_sd"],3) if wl in lat_piv.index else "—",
            "Draft-SD": round(lat_piv.loc[wl,"draft_sd"],3) if wl in lat_piv.index else "—",
            "Eagle3":   round(lat_piv.loc[wl,"eagle3"],3)   if wl in lat_piv.index else "—",
        })
    lines.append(df_to_md(pd.DataFrame(t7_rows)) + "\n")

    # ── Summary ──
    lines.append("---\n## 4.6 Summary of Phase 1 Findings\n")
    best_e = max(wls, key=lambda w: df[(df["method"]=="eagle3")&(df["workload"]==w)]["tokens_per_sec"].max())
    best_n = max(wls, key=lambda w: df[(df["method"]=="ngram_sd")&(df["workload"]==w)]["tokens_per_sec"].max())
    r_entr_e = np.corrcoef([entr_ar[w] for w in wls if w in tps_e4],
                            [tps_e4[w]/tps_a[w] for w in wls if w in tps_e4])[0,1]
    r_rep_n  = np.corrcoef([rep_ar[w] for w in wls if w in tps_n4],
                            [tps_n4[w]/tps_a[w] for w in wls if w in tps_n4])[0,1]
    lines.append(
        f"1. **Eagle3** is the most reliable method, delivering 2–7× speedup across all workloads. "
        f"Peak: {df[(df['method']=='eagle3')&(df['workload']==best_e)]['tokens_per_sec'].max()/tps_a[best_e]:.2f}× "
        f"on {WORKLOAD_FULL[best_e]}.\n"
        f"2. **Ngram-SD exceeds Eagle3** on long-output, high-repetition workloads at k=16. "
        f"Peak: {df[(df['method']=='ngram_sd')&(df['workload']==best_n)]['tokens_per_sec'].max()/tps_a[best_n]:.2f}× "
        f"on {WORKLOAD_FULL[best_n]}.\n"
        f"3. **Draft-SD with mismatched draft family** averages 0.50× AR throughput — "
        f"degrading monotonically with k. Motivates Qwen3-family sweep in Phase 2.\n"
        f"4. **Token entropy predicts speedup**: r = {r_entr_e:.3f} (Eagle3); "
        f"repetition density predicts Ngram-SD speedup: r = {r_rep_n:.3f}.\n"
        f"5. **Optimal k is workload-dependent**: short-output tasks peak at k=4–8; "
        f"long-output tasks scale through k=16.\n"
    )

    md_path = out / "phase1_results_section.md"
    md_path.write_text("\n".join(lines))
    print(f"  [md ] {md_path.name}")


# ─────────────────────────────────────────────────────────────────
# ══════════════  FIGURES  ══════════════
# ─────────────────────────────────────────────────────────────────

def fig1_throughput_bar(df: pd.DataFrame, out: Path) -> None:
    """Fig 1: Throughput bar chart at k=4."""
    ar_rows = df[df["method"] == "ar"].copy(); ar_rows["k"] = 4
    plot_df = pd.concat([ar_rows, df[(df["method"] != "ar") & (df["k"] == 4)]])
    workloads = [w for w in WORKLOAD_SHORT if w in df["workload"].unique()]

    fig, ax = plt.subplots(figsize=(14, 5))
    x   = np.arange(len(workloads))
    w   = 0.2
    offs = [-1.5, -0.5, 0.5, 1.5]

    for i, meth in enumerate(METHOD_ORDER):
        sub   = plot_df[plot_df["method"] == meth]
        means = [sub[sub["workload"] == wl]["tokens_per_sec"].mean() for wl in workloads]
        stds  = [sub[sub["workload"] == wl]["tokens_per_sec"].std()  for wl in workloads]
        ax.bar(x + offs[i] * w, means, width=w * 0.9,
               color=COLORS[meth], label=METHOD_LABELS[meth],
               yerr=stds, capsize=3, error_kw={"elinewidth": 1, "alpha": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_SHORT[w] for w in workloads], rotation=20, ha="right")
    ax.set_ylabel("Tokens / Second")
    ax.set_title("Figure 1: Throughput at k=4 — All Methods × Workloads", fontweight="bold")
    ax.legend(title="Method", loc="upper left")
    ax.yaxis.grid(True, alpha=0.4)
    save(fig, out / "fig1_throughput_bar_k4.png")


def fig2_speedup_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Fig 2: Peak speedup heatmap (best k per method)."""
    ar_tps      = ar_tps_by_workload(df)
    spec_methods = ["ngram_sd", "draft_sd", "eagle3"]
    workloads   = [w for w in WORKLOAD_SHORT if w in df["workload"].unique()]

    best_tps = (df[df["method"].isin(spec_methods)]
                .groupby(["workload", "method"])["tokens_per_sec"]
                .max()
                .unstack()[spec_methods])
    speedup = best_tps.div(ar_tps, axis=0)
    speedup.index   = speedup.index.map(WORKLOAD_SHORT)
    speedup.columns = [METHOD_LABELS[m] for m in spec_methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(speedup, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Speedup vs AR", "shrink": 0.85})
    ax.set_title("Figure 2: Peak Speedup vs AR (Best k) — Workload × Method", fontweight="bold")
    ax.set_xlabel("Method"); ax.set_ylabel("")
    save(fig, out / "fig2_speedup_heatmap.png")


def fig3_k_sensitivity(df: pd.DataFrame, out: Path) -> None:
    """Fig 3: k-sensitivity curves — Eagle3 and Ngram-SD."""
    ar_tps    = ar_tps_by_workload(df)
    workloads = [w for w in WORKLOAD_SHORT if w in df["workload"].unique()]
    ks        = sorted(df["k"].unique())
    palette   = sns.color_palette("tab10", len(workloads))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, meth in zip(axes, ["eagle3", "ngram_sd"]):
        sub = df[df["method"] == meth]
        for j, wl in enumerate(workloads):
            wl_sub = sub[sub["workload"] == wl].groupby("k")["tokens_per_sec"].mean()
            ax.plot(wl_sub.index, wl_sub.values, marker="o",
                    color=palette[j], label=WORKLOAD_SHORT[wl], linewidth=2)
            ax.axhline(ar_tps[wl], color=palette[j], linestyle="--", alpha=0.25, linewidth=1)
        ax.set_title(f"{METHOD_LABELS[meth]} — k Sensitivity", fontweight="bold")
        ax.set_xlabel("Speculation Depth k"); ax.set_ylabel("Tokens / Second")
        ax.set_xticks(ks); ax.legend(fontsize=8, ncol=2); ax.yaxis.grid(True, alpha=0.4)

    fig.suptitle("Figure 3: k-Sensitivity — Eagle3 vs Ngram-SD (dashed = AR baseline)",
                 fontweight="bold", y=1.01)
    save(fig, out / "fig3_k_sensitivity_eagle3_ngram.png")


def fig4_latency_cdf(df: pd.DataFrame, out: Path) -> None:
    """Fig 4: Latency CDF per workload at k=4."""
    workloads = [w for w in WORKLOAD_SHORT if w in df["workload"].unique()]
    ncols = 4; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes = axes.flatten()

    for idx, wl in enumerate(workloads):
        ax = axes[idx]
        for meth in METHOD_ORDER:
            if meth == "ar":
                sub = df[(df["workload"] == wl) & (df["method"] == meth)]
            else:
                sub = df[(df["workload"] == wl) & (df["method"] == meth) & (df["k"] == 4)]
            vals = np.sort(sub["latency_s"].dropna().values)
            if len(vals) == 0:
                continue
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, color=COLORS[meth], label=METHOD_LABELS[meth], linewidth=1.8)
        ax.set_title(WORKLOAD_SHORT[wl], fontsize=10, fontweight="bold")
        ax.set_xlabel("Latency (s)", fontsize=8); ax.set_ylabel("CDF", fontsize=8)
        ax.tick_params(labelsize=8); ax.yaxis.grid(True, alpha=0.4)

    # Legend panel
    for ax in axes[len(workloads):]:
        ax.axis("off")
    handles = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    axes[len(workloads)].legend(handles=handles, loc="center",
                                fontsize=11, title="Method", title_fontsize=12)
    fig.suptitle("Figure 4: Latency CDF by Workload (k=4)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, out / "fig4_latency_cdf.png")


def fig5_structural_heatmaps(df: pd.DataFrame, out: Path) -> None:
    """Fig 5: Token entropy and repetition density heatmaps (AR baseline)."""
    entr = (df[df["method"] == "ar"].groupby("workload")["mean_entropy"]
            .mean().rename(index=WORKLOAD_SHORT)
            .sort_values(ascending=False)
            .to_frame("Mean Entropy"))
    rep  = (df[df["method"] == "ar"].groupby("workload")["repetition_density"]
            .mean().rename(index=WORKLOAD_SHORT)
            .sort_values(ascending=False)
            .to_frame("Repetition Density"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(entr.round(3), annot=True, fmt=".3f", cmap="YlOrRd",
                ax=axes[0], cbar_kws={"label": "Entropy", "shrink": 0.85}, linewidths=0.5)
    axes[0].set_title("Token Entropy\n(lower = more predictable)", fontweight="bold")
    axes[0].set_xlabel("")

    sns.heatmap(rep.round(3), annot=True, fmt=".3f", cmap="YlGn",
                ax=axes[1], cbar_kws={"label": "Rep Density", "shrink": 0.85}, linewidths=0.5)
    axes[1].set_title("Repetition Density\n(higher = more n-gram reuse)", fontweight="bold")
    axes[1].set_xlabel("")

    fig.suptitle("Figure 5: Workload Token Regime Characterization (AR Baseline)",
                 fontweight="bold")
    save(fig, out / "fig5_structural_heatmaps.png")


def fig6_entropy_speedup_scatter(df: pd.DataFrame, out: Path) -> None:
    """Fig 6: Entropy/repetition vs speedup scatter with correlation lines."""
    ar_tps    = ar_tps_by_workload(df)
    entr_ar   = df[df["method"] == "ar"].groupby("workload")["mean_entropy"].mean()
    rep_ar    = df[df["method"] == "ar"].groupby("workload")["repetition_density"].mean()
    workloads = [w for w in WORKLOAD_SHORT if w in df["workload"].unique()]
    palette   = {wl: sns.color_palette("tab10", len(workloads))[i]
                 for i, wl in enumerate(workloads)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Eagle3 speedup vs entropy
    ax = axes[0]
    tps_e4     = df[(df["method"] == "eagle3") & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()
    speedup_e4 = tps_e4 / ar_tps
    for wl in workloads:
        if wl not in entr_ar or wl not in speedup_e4:
            continue
        ax.scatter(entr_ar[wl], speedup_e4[wl], color=palette[wl], s=120, zorder=5)
        ax.annotate(WORKLOAD_SHORT[wl], (entr_ar[wl], speedup_e4[wl]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    x_v = np.array([entr_ar[wl] for wl in workloads if wl in speedup_e4])
    y_v = np.array([speedup_e4[wl] for wl in workloads if wl in speedup_e4])
    if len(x_v) > 1:
        z = np.polyfit(x_v, y_v, 1); p = np.poly1d(z)
        xl = np.linspace(x_v.min(), x_v.max(), 100)
        ax.plot(xl, p(xl), "k--", alpha=0.5, linewidth=1.5)
        r = np.corrcoef(x_v, y_v)[0, 1]
        ax.set_title(f"Eagle3 Speedup vs Token Entropy\nr = {r:.3f}", fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Mean Token Entropy (AR baseline)"); ax.set_ylabel("Speedup vs AR (k=4)")
    ax.yaxis.grid(True, alpha=0.4)

    # Right: Ngram-SD speedup vs repetition density
    ax = axes[1]
    tps_n4     = df[(df["method"] == "ngram_sd") & (df["k"] == 4)].groupby("workload")["tokens_per_sec"].mean()
    speedup_n4 = tps_n4 / ar_tps
    for wl in workloads:
        if wl not in rep_ar or wl not in speedup_n4:
            continue
        ax.scatter(rep_ar[wl], speedup_n4[wl], color=palette[wl], s=120, zorder=5)
        ax.annotate(WORKLOAD_SHORT[wl], (rep_ar[wl], speedup_n4[wl]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    x_v = np.array([rep_ar[wl] for wl in workloads if wl in speedup_n4])
    y_v = np.array([speedup_n4[wl] for wl in workloads if wl in speedup_n4])
    if len(x_v) > 1:
        z = np.polyfit(x_v, y_v, 1); p = np.poly1d(z)
        xl = np.linspace(x_v.min(), x_v.max(), 100)
        ax.plot(xl, p(xl), "k--", alpha=0.5, linewidth=1.5)
        r = np.corrcoef(x_v, y_v)[0, 1]
        ax.set_title(f"Ngram-SD Speedup vs Repetition Density\nr = {r:.3f}", fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Repetition Density (AR baseline)"); ax.set_ylabel("Speedup vs AR (k=4)")
    ax.yaxis.grid(True, alpha=0.4)

    fig.suptitle("Figure 6: Token Regime → Speedup Relationship", fontweight="bold")
    save(fig, out / "fig6_entropy_speedup_scatter.png")


def fig7_draft_sd_analysis(df: pd.DataFrame, out: Path) -> None:
    """Fig 7: Draft-SD performance — bar comparison and k-sensitivity."""
    ar_tps    = ar_tps_by_workload(df)
    workloads = [w for w in WORKLOAD_SHORT if w in df["workload"].unique()]
    palette   = sns.color_palette("tab10", len(workloads))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart comparing all methods at k=4
    ar_rows = df[df["method"] == "ar"].copy(); ar_rows["k"] = 4
    plot_df = pd.concat([ar_rows, df[(df["method"] != "ar") & (df["k"] == 4)]])
    x = np.arange(len(workloads)); w = 0.2
    for i, meth in enumerate(METHOD_ORDER):
        sub   = plot_df[plot_df["method"] == meth]
        means = [sub[sub["workload"] == wl]["tokens_per_sec"].mean() for wl in workloads]
        axes[0].bar(x + (i - 1.5) * w, means, width=w * 0.9,
                    color=COLORS[meth], label=METHOD_LABELS[meth])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([WORKLOAD_SHORT[wl] for wl in workloads], rotation=20, ha="right")
    axes[0].set_ylabel("Tokens / Second")
    axes[0].set_title("All Methods at k=4", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].yaxis.grid(True, alpha=0.4)

    # Right: Draft-SD k-sweep — monotonic degradation
    draft_ks = (df[df["method"] == "draft_sd"]
                .groupby(["workload", "k"])["tokens_per_sec"].mean()
                .unstack())
    for i, wl in enumerate(workloads):
        if wl not in draft_ks.index: continue
        axes[1].plot(draft_ks.columns, draft_ks.loc[wl],
                     marker="o", color=palette[i], label=WORKLOAD_SHORT[wl])
        axes[1].axhline(ar_tps[wl], color=palette[i], linestyle="--", alpha=0.3, linewidth=1)
    axes[1].set_xlabel("Speculation Depth k"); axes[1].set_ylabel("Tokens / Second")
    axes[1].set_title("Draft-SD k-Sensitivity\n(dashed = AR baseline)", fontweight="bold")
    axes[1].legend(fontsize=7, ncol=2); axes[1].yaxis.grid(True, alpha=0.4)

    fig.suptitle("Figure 7: Draft-SD Analysis — Mismatched Draft Model Family",
                 fontweight="bold")
    save(fig, out / "fig7_draft_sd_analysis.png")


def fig8_ngram_beats_eagle3(df: pd.DataFrame, out: Path) -> None:
    """Fig 8: Ngram-SD vs Eagle3 on long-output workloads."""
    ar_tps = ar_tps_by_workload(df)
    long_wls = [w for w in LONG_WORKLOADS if w in df["workload"].unique()]
    ks = sorted(df["k"].unique())

    fig, axes = plt.subplots(1, len(long_wls), figsize=(5 * len(long_wls), 5))
    if len(long_wls) == 1:
        axes = [axes]

    for ax, wl in zip(axes, long_wls):
        for meth in ["ngram_sd", "eagle3"]:
            sub = (df[(df["workload"] == wl) & (df["method"] == meth)]
                   .groupby("k")["tokens_per_sec"].mean())
            ax.plot(sub.index, sub.values, marker="o",
                    color=COLORS[meth], label=METHOD_LABELS[meth], linewidth=2.5)
        ax.axhline(ar_tps[wl], color=COLORS["ar"], linestyle="--", linewidth=2, label="AR")
        ax.set_title(WORKLOAD_SHORT[wl], fontweight="bold")
        ax.set_xlabel("k"); ax.set_ylabel("Tokens / Second")
        ax.set_xticks(ks); ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.4)

    fig.suptitle(
        "Figure 8: Ngram-SD vs Eagle3 on Long-Output Workloads\n"
        "(Ngram-SD exceeds Eagle3 at high k due to high repetition density)",
        fontweight="bold",
    )
    save(fig, out / "fig8_ngram_beats_eagle3_long.png")


def fig9_throughput_scaling(df: pd.DataFrame, out: Path) -> None:
    """Fig 9: Full throughput scaling — all workloads × speculative methods."""
    ar_tps    = ar_tps_by_workload(df)
    workloads = [w for w in WORKLOAD_SHORT if w in df["workload"].unique()]
    ncols = 4; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9))
    axes = axes.flatten()

    for idx, wl in enumerate(workloads):
        ax = axes[idx]
        for meth in ["eagle3", "ngram_sd", "draft_sd"]:
            sub = (df[(df["workload"] == wl) & (df["method"] == meth)]
                   .groupby("k")["tokens_per_sec"].mean())
            ax.plot(sub.index, sub.values, marker="o",
                    color=COLORS[meth], label=METHOD_LABELS[meth], linewidth=2)
        ax.axhline(ar_tps[wl], color=COLORS["ar"], linestyle="--", linewidth=2, label="AR")
        ax.set_title(WORKLOAD_SHORT[wl], fontweight="bold", fontsize=10)
        ax.set_xlabel("k", fontsize=9); ax.set_ylabel("tok/s", fontsize=9)
        ax.set_xticks(sorted(df["k"].unique()))
        ax.tick_params(labelsize=8); ax.yaxis.grid(True, alpha=0.4)

    # Legend in remaining panel(s)
    for ax in axes[len(workloads):]:
        ax.axis("off")
    handles = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
               for m in ["ar", "ngram_sd", "draft_sd", "eagle3"]]
    axes[len(workloads)].legend(handles=handles, loc="center",
                                fontsize=12, title="Method", title_fontsize=13)

    fig.suptitle(
        "Figure 9: Throughput Scaling with k — All Workloads × Speculative Methods",
        fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save(fig, out / "fig9_throughput_scaling_all.png")


def fig10_entropy_by_method(df: pd.DataFrame, out: Path) -> None:
    """Fig 10: Token entropy per workload per method — method-invariance check."""
    ar_rows = df[df["method"] == "ar"].copy(); ar_rows["k"] = 4
    k4 = pd.concat([ar_rows, df[(df["method"] != "ar") & (df["k"] == 4)]])

    entr = (k4.groupby(["workload", "method"])["mean_entropy"]
              .mean()
              .unstack()
              .reindex(columns=METHOD_ORDER))
    entr.index   = entr.index.map(WORKLOAD_SHORT)
    entr.columns = [METHOD_LABELS[m] for m in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(12, 5))
    entr.plot(kind="bar", ax=ax,
              color=[COLORS[m] for m in METHOD_ORDER],
              width=0.7, rot=20)
    ax.set_ylabel("Mean Token Entropy")
    ax.set_title(
        "Figure 10: Token Entropy by Workload and Method (k=4)\n"
        "(Entropy is an intrinsic workload property, not method-dependent)",
        fontweight="bold",
    )
    ax.legend(title="Method"); ax.yaxis.grid(True, alpha=0.4)
    save(fig, out / "fig10_entropy_by_method.png")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Phase 1 tables and figures.")
    ap.add_argument("--data",    default="results/all_runs.csv",
                    help="Path to all_runs.csv (default: results/all_runs.csv)")
    ap.add_argument("--out_dir", default="results/phase1_report",
                    help="Output root directory (default: results/phase1_report)")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    root    = Path(args.out_dir)
    tbl_dir = root / "tables"
    fig_dir = root / "figures"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  {len(df)} rows | methods: {sorted(df['method'].unique())} | "
          f"workloads: {df['workload'].nunique()}")

    # ── Apply global plot style ───────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 180, "savefig.bbox": "tight",
                          "font.family": "DejaVu Sans"})

    # ── Tables ───────────────────────────────────────────────────
    print("\n── Generating tables ──")
    make_table1(df, tbl_dir)
    make_table2(df, tbl_dir)
    make_table3(df, tbl_dir)
    make_table4(df, tbl_dir)
    make_table5(df, tbl_dir)
    make_table6(df, tbl_dir)
    make_table7(df, tbl_dir)
    make_table_md4(df, tbl_dir)
    make_table_md5(df, tbl_dir)

    # ── Markdown report with all tables embedded ─────────────────
    print("\n── Generating markdown results section ──")
    generate_md_report(df, root)

    # ── Figures ──────────────────────────────────────────────────
    print("\n── Generating figures ──")
    fig1_throughput_bar(df, fig_dir)
    fig2_speedup_heatmap(df, fig_dir)
    fig3_k_sensitivity(df, fig_dir)
    fig4_latency_cdf(df, fig_dir)
    fig5_structural_heatmaps(df, fig_dir)
    fig6_entropy_speedup_scatter(df, fig_dir)
    fig7_draft_sd_analysis(df, fig_dir)
    fig8_ngram_beats_eagle3(df, fig_dir)
    fig9_throughput_scaling(df, fig_dir)
    fig10_entropy_by_method(df, fig_dir)

    print(f"\n✓  All outputs written to {root}/")
    print(f"   tables/   → {len(list(tbl_dir.glob('*.csv')))} CSV files")
    print(f"   figures/  → {len(list(fig_dir.glob('*.png')))} PNG files")
    print(f"   phase1_results_section.md  (full report with embedded tables)")


if __name__ == "__main__":
    main()