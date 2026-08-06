#!/usr/bin/env python
"""
plot_staleness.py — figures from the staleness aggregate outputs
================================================================
Consumes the CSVs from aggregate_staleness.py and produces paper figures.
Robust to: missing output-KL (falls back to weight distance), Tülu's
non-numeric / nan drift distance (falls back to trajectory order), and any
family/study/method being absent.

Figures (PNG, 150 dpi) -> --out_dir:
  fig1_drift_curves.png        acceptance vs drift, line/drafter, panel/family (S1)
  fig2_slopes.png              staleness-sensitivity slope per family x drafter
  fig3_retention_heatmap.png   acceptance retention by drift level x drafter
  fig4_temperature.png         acceptance vs drift, line/temperature (S2)
  fig5_position_decay.png      per-position accept rate vs drift
  fig6_tulu_trajectory.png     acceptance along real Base->SFT->DPO->RLVR

Usage:
  python plot_staleness.py --in_dir results/staleness --out_dir results/staleness/figs
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_STYLE = {
    "ngram_sd": dict(color="#2a9d8f", marker="o", label="n-gram"),
    "draft_sd": dict(color="#e9c46a", marker="s", label="draft model"),
    "eagle3":   dict(color="#e76f51", marker="^", label="EAGLE-3"),
}
TULU_ORDER = ["Llama-3.1-8B", "Tulu-3-8B-SFT", "Tulu-3-8B-DPO", "Llama-3.1-Tulu-3-8B"]


def _read(in_dir, name):
    p = Path(in_dir) / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _x_axis(df):
    if "mean_token_kl" in df.columns and df["mean_token_kl"].notna().any():
        return "mean_token_kl", "output KL(drift || ref)"
    return "rel_weight_dist", "relative weight drift"


def _curve(ax, sub, xcol):
    for method, g in sub.groupby("method"):
        gg = (g.groupby(xcol, dropna=True)["acceptance_rate"]
                .mean().reset_index().sort_values(xcol))
        if gg.empty:
            continue
        st = METHOD_STYLE.get(method, dict(color="gray", marker="o", label=method))
        ax.plot(gg[xcol], gg["acceptance_rate"], marker=st["marker"],
                color=st["color"], label=st["label"], lw=2, ms=6)


def fig_drift_curves(cells, out_dir):
    s1 = cells[cells.study == "S1_core_drift_curve"]
    if s1.empty:
        return
    s1 = s1[s1.k == s1.k.min()]
    fams = [f for f in ["qwen_interp", "llama_interp"] if f in set(s1.family)]
    if not fams:
        return
    xcol, xlabel = _x_axis(s1)
    fig, axes = plt.subplots(1, len(fams), figsize=(6 * len(fams), 4.5), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        _curve(ax, s1[s1.family == fam], xcol)
        ax.set_title(fam.replace("_", " "))
        ax.set_xlabel(xlabel); ax.set_ylabel("acceptance rate")
        ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Draft acceptance vs target drift (lower x = fresher)", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "fig1_drift_curves.png",
                                    dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_slopes(slopes, out_dir):
    if slopes.empty:
        return
    s = slopes[(slopes.study == "S1_core_drift_curve") & (slopes["mode"] == "interp")]
    if "x" in s.columns:
        s = s[s.x == "rel_weight_dist"] if (s.x == "rel_weight_dist").any() else s
    if s.empty:
        return
    s = s.sort_values(["family", "slope"])
    labels = [f"{r.family.split('_')[0]}\n{METHOD_STYLE.get(r.method,{}).get('label', r.method)}"
              for r in s.itertuples()]
    colors = [METHOD_STYLE.get(m, {}).get("color", "gray") for m in s.method]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(s)), 4.2))
    ax.bar(range(len(s)), s.slope, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(s))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("acceptance-vs-drift slope")
    ax.set_title("Staleness sensitivity (more negative = more fragile)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "fig2_slopes.png",
                                    dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_retention_heatmap(cells, out_dir):
    s1 = cells[(cells.study == "S1_core_drift_curve") & cells.acceptance_retention.notna()]
    fams = [f for f in ["qwen_interp", "llama_interp"] if f in set(s1.family)]
    if not fams:
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        sub = s1[s1.family == fam]
        piv = sub.pivot_table(index="method", columns="param",
                              values="acceptance_retention", aggfunc="mean")
        if piv.empty:
            continue
        piv = piv[sorted(piv.columns, reverse=True)]
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.05)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([METHOD_STYLE.get(m, {}).get("label", m) for m in piv.index])
        ax.set_xlabel("interp alpha (1=fresh)"); ax.set_title(fam.replace("_", " "))
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label="acceptance retention")
    fig.suptitle("Acceptance retention vs fresh checkpoint", y=1.02)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "fig3_retention_heatmap.png",
                                    dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_temperature(cells, out_dir):
    s2 = cells[cells.study == "S2_temperature_interaction"]
    if s2.empty:
        return
    xcol, xlabel = _x_axis(s2)
    methods = [m for m in ["ngram_sd", "draft_sd", "eagle3"] if m in set(s2.method)]
    fams = list(s2.family.unique())
    fig, axes = plt.subplots(len(fams), len(methods),
                             figsize=(4 * len(methods), 3.3 * len(fams)), squeeze=False)
    cmap = plt.get_cmap("viridis")
    temps = sorted(s2.temperature.unique())
    tmax = max(temps) if temps else 1
    for r, fam in enumerate(fams):
        for c, method in enumerate(methods):
            ax = axes[r][c]
            sub = s2[(s2.family == fam) & (s2.method == method)]
            for t in temps:
                gg = (sub[sub.temperature == t].groupby(xcol)["acceptance_rate"]
                      .mean().reset_index().sort_values(xcol))
                if gg.empty:
                    continue
                ax.plot(gg[xcol], gg["acceptance_rate"], marker="o", ms=4,
                        color=cmap(t / tmax if tmax else 0), label=f"T={t:g}")
            if r == 0:
                ax.set_title(METHOD_STYLE.get(method, {}).get("label", method))
            if c == 0:
                ax.set_ylabel(f"{fam.split('_')[0]}\nacceptance")
            ax.set_xlabel(xlabel); ax.grid(alpha=0.3)
            if r == 0 and c == len(methods) - 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Drift x temperature interaction", y=1.01)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "fig4_temperature.png",
                                    dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_position_decay(pos, out_dir):
    if pos.empty:
        return
    p = pos[pos.study == "S1_core_drift_curve"] if "study" in pos.columns else pos
    fams = list(p.family.unique()) if "family" in p.columns else [None]
    methods = [m for m in ["draft_sd", "eagle3"] if m in set(p.method)] or list(p.method.unique())
    fig, axes = plt.subplots(len(fams), len(methods),
                             figsize=(4 * len(methods), 3.2 * len(fams)), squeeze=False)
    for r, fam in enumerate(fams):
        for c, method in enumerate(methods):
            ax = axes[r][c]
            sub = p[p.method == method]
            if fam is not None:
                sub = sub[sub.family == fam]
            drifts = sorted(sub["rel_weight_dist"].dropna().unique())
            cmap = plt.get_cmap("plasma")
            for i, dval in enumerate(drifts):
                gg = (sub[sub.rel_weight_dist == dval].groupby("position")["accept_rate"]
                      .mean().reset_index().sort_values("position"))
                if gg.empty:
                    continue
                ax.plot(gg["position"], gg["accept_rate"], marker="o", ms=4,
                        color=cmap(i / max(len(drifts) - 1, 1)), label=f"drift={dval:.3g}")
            if r == 0:
                ax.set_title(METHOD_STYLE.get(method, {}).get("label", method))
            if c == 0:
                ax.set_ylabel(f"{fam}\naccept rate" if fam else "accept rate")
            ax.set_xlabel("draft position"); ax.grid(alpha=0.3)
            if r == 0 and c == len(methods) - 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Per-position acceptance decay under drift", y=1.01)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "fig5_position_decay.png",
                                    dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_tulu(cells, out_dir):
    t = cells[cells.family == "tulu_trajectory"].copy()
    if t.empty:
        return
    t = t[t.study.isin(["S1_core_drift_curve", "S4_real_trajectory_focus"])]
    if t.empty:
        return

    def order_key(tag):
        for i, name in enumerate(TULU_ORDER):
            if name in str(tag):
                return i
        return len(TULU_ORDER)
    t["traj"] = t["drift_tag"].map(order_key)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, g in t.groupby("method"):
        gg = g.groupby("traj")["acceptance_rate"].mean().reset_index().sort_values("traj")
        st = METHOD_STYLE.get(method, dict(color="gray", marker="o", label=method))
        ax.plot(gg["traj"], gg["acceptance_rate"], marker=st["marker"],
                color=st["color"], label=st["label"], lw=2, ms=7)
    labels = ["Base", "SFT", "DPO", "RLVR"]
    present = sorted(t["traj"].unique())
    ax.set_xticks(present)
    ax.set_xticklabels([labels[i] if i < len(labels) else "?" for i in present])
    ax.set_xlabel("real post-training trajectory ->")
    ax.set_ylabel("acceptance rate")
    ax.set_title("Acceptance along the real Tulu RLHF path (Base->SFT->DPO->RLVR)")
    ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "fig6_tulu_trajectory.png",
                                    dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/staleness")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir or (Path(args.in_dir) / "figs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = _read(args.in_dir, "staleness_cells.csv")
    slopes = _read(args.in_dir, "staleness_slopes.csv")
    pos = _read(args.in_dir, "staleness_position_decay.csv")
    if cells.empty:
        raise SystemExit(f"No staleness_cells.csv in {args.in_dir} — run aggregate first.")

    for fn, a in [(fig_drift_curves, (cells, out_dir)),
                  (fig_slopes, (slopes, out_dir)),
                  (fig_retention_heatmap, (cells, out_dir)),
                  (fig_temperature, (cells, out_dir)),
                  (fig_position_decay, (pos, out_dir)),
                  (fig_tulu, (cells, out_dir))]:
        try:
            fn(*a)
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {type(e).__name__}: {e}")
    pngs = sorted(p.name for p in out_dir.glob("*.png"))
    print(f"Wrote {len(pngs)} figure(s) to {out_dir}:")
    for p in pngs:
        print("  ", p)


if __name__ == "__main__":
    main()
