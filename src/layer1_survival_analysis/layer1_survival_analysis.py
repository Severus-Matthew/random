#!/usr/bin/env python
"""Layer 1 survival reinterpretation for APEX.

This script consumes Phase-2 `results/all_runs.csv` and reconstructs
request-level accepted-length survival profiles from vLLM positional counters.

Important convention for the existing results:
- `accept_rate_pos_j` / `survival_pos_j` are interpreted as survival S_j,
  i.e. the probability that positions 0..j all survived.
- Conditional acceptance is p_0=S_0 and p_j=S_j/S_{j-1} for j>0.
- h_j=1-p_j.
- PMF: P(L=0)=1-S_0; P(L=j)=S_{j-1}-S_j for j>=1.
- If all k positions are observed, P(L=k)=S_{k-1}. If only a prefix is
  observed, the final observed S is a tail mass P(L >= n_observed), not P(L=n).

The script deliberately treats AR separately: AR is used to compute speedup but
is excluded from survival-specific plots/correlations because it has no draft
survival curve.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for Layer 1 figures") from exc

_POS_RE = re.compile(r"(?:accept_rate_pos_|survival_pos_)(\d+)$")


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _collect_survival(record: Mapping[str, Any], *, max_depth: int | None) -> dict[int, float]:
    pos: dict[int, float] = {}
    for key, value in record.items():
        m = _POS_RE.match(str(key))
        if not m:
            continue
        j = int(m.group(1))
        if max_depth is not None and j >= max_depth:
            continue
        val = _finite_float(value)
        if val is None:
            continue
        pos[j] = _clip01(val)
    # Keep only contiguous prefix. Missing middle positions make later positions ambiguous.
    if not pos:
        return {}
    out: dict[int, float] = {}
    for j in range(max(pos) + 1):
        if j not in pos:
            break
        out[j] = pos[j]
    return out


def _enrich_one(record: Mapping[str, Any], *, max_depth: int | None) -> dict[str, Any]:
    out = dict(record)
    method = str(out.get("method", "")).lower()

    # Remove old derived columns if the input came from a previous Layer-1 run.
    for key in list(out.keys()):
        if key.startswith(("survival_pos_", "conditional_accept_pos_", "hazard_pos_", "pmf_L_eq_")):
            # Keep survival_pos if source has no accept_rate_pos. We handle this below by collecting first.
            pass

    survival = _collect_survival(out, max_depth=max_depth)
    k_val = _finite_float(out.get("k", out.get("draft_len")))
    k = int(k_val) if k_val is not None and k_val > 0 else None

    if method == "ar" or not survival:
        out.update({
            "apex_observed_positions": 0.0,
            "apex_expected_len_survival_observed": np.nan,
            "apex_expected_len_counter": np.nan if method == "ar" else _finite_float(out.get("accepted_per_draft")) or np.nan,
            "apex_tail_expected_missing": np.nan,
            "apex_observed_tail_mass_L_ge_n": np.nan,
            "apex_pmf_mass_observed_prefix_plus_tail": np.nan,
            "apex_survival_monotonic_violations": np.nan,
            "apex_trace_level": "ar_no_survival" if method == "ar" else "request_aggregate",
            "apex_strict_block_trace": False,
            "apex_layer0_schema_version": "0.3-request-survival-fixed",
        })
        return out

    vals = [_clip01(survival[j]) for j in sorted(survival)]
    n = len(vals)
    out["apex_observed_positions"] = float(n)
    out["apex_expected_len_survival_observed"] = float(sum(vals))

    prev = 1.0
    violations = 0
    for j, sj in enumerate(vals):
        if sj > prev + 1e-9:
            violations += 1
        pj = _clip01(sj / prev) if prev > 1e-12 else 0.0
        out[f"survival_pos_{j}"] = sj
        out[f"conditional_accept_pos_{j}"] = pj
        out[f"hazard_pos_{j}"] = 1.0 - pj
        if j == 0:
            out["pmf_L_eq_0"] = 1.0 - sj
        else:
            out[f"pmf_L_eq_{j}"] = max(0.0, prev - sj)
        prev = sj

    out["apex_survival_monotonic_violations"] = float(violations)
    out["apex_observed_tail_mass_L_ge_n"] = vals[-1]
    out["apex_pmf_mass_observed_prefix_plus_tail"] = 1.0

    # Only if every position 0..k-1 is observed is the tail exactly P(L=k).
    if k is not None and n >= k:
        out[f"pmf_L_eq_{k}"] = vals[k - 1]
        out["apex_pmf_mass_for_depth"] = sum(float(out.get(f"pmf_L_eq_{j}", 0.0)) for j in range(k + 1))
    else:
        out["apex_pmf_mass_for_depth"] = np.nan

    accepted_per_draft = _finite_float(out.get("accepted_per_draft"))
    accepted_total = _finite_float(out.get("accepted_tokens_total", out.get("accepted_tokens")))
    num_drafts = _finite_float(out.get("num_drafts"))
    if accepted_per_draft is not None:
        out["apex_expected_len_counter"] = accepted_per_draft
    elif accepted_total is not None and num_drafts is not None and num_drafts > 0:
        out["apex_expected_len_counter"] = accepted_total / num_drafts
    else:
        out["apex_expected_len_counter"] = np.nan

    ctr = _finite_float(out.get("apex_expected_len_counter"))
    obs = _finite_float(out.get("apex_expected_len_survival_observed"))
    out["apex_tail_expected_missing"] = (ctr - obs) if ctr is not None and obs is not None else np.nan
    out["apex_trace_level"] = "request_aggregate"
    out["apex_strict_block_trace"] = False
    out["apex_layer0_schema_version"] = "0.3-request-survival-fixed"
    return out


def _add_speedup_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    if "speedup_vs_ar" in df.columns and df["speedup_vs_ar"].notna().any():
        return df
    if "tokens_per_sec" not in df.columns or "method" not in df.columns:
        df["speedup_vs_ar"] = np.nan
        return df
    keys = [c for c in ["workload", "source_dataset", "temperature", "max_prompt_tokens", "num_turns"] if c in df.columns]
    if not keys and "workload" in df.columns:
        keys = ["workload"]
    ar = df[df["method"].astype(str).str.lower() == "ar"].copy()
    if ar.empty or not keys:
        df["speedup_vs_ar"] = np.nan
        return df
    ar_tps = ar.groupby(keys, dropna=False)["tokens_per_sec"].mean().rename("_ar_tps").reset_index()
    out = df.merge(ar_tps, on=keys, how="left")
    out["speedup_vs_ar"] = out["tokens_per_sec"] / out["_ar_tps"].replace(0, np.nan)
    return out.drop(columns=["_ar_tps"], errors="ignore")


def enrich_dataframe(df: pd.DataFrame, *, max_depth: int | None) -> pd.DataFrame:
    enriched = pd.DataFrame([_enrich_one(rec, max_depth=max_depth) for rec in df.to_dict(orient="records")])
    return _add_speedup_if_missing(enriched)


def survival_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "method" in df.columns:
        df = df[df["method"].astype(str).str.lower() != "ar"].copy()
    if "apex_observed_positions" in df.columns:
        df = df[df["apex_observed_positions"].fillna(0) > 0].copy()
    return df


def group_survival(df: pd.DataFrame) -> pd.DataFrame:
    df = survival_subset(df)
    group_cols = [c for c in ["workload", "method", "k", "temperature", "ngram_lookup_min", "ngram_lookup_max"] if c in df.columns]
    if not group_cols:
        group_cols = ["method"] if "method" in df.columns else []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    wanted_prefixes = ("survival_pos_", "conditional_accept_pos_", "hazard_pos_", "pmf_L_eq_")
    wanted = [c for c in numeric_cols if c.startswith(wanted_prefixes)]
    extra = [
        "tokens_per_sec", "speedup_vs_ar", "acceptance_rate", "accepted_per_draft",
        "apex_expected_len_survival_observed", "apex_expected_len_counter",
        "apex_tail_expected_missing", "mean_entropy", "repetition_density",
        "first_rejection_pos", "oracle_k_estimated", "latency_s",
        "apex_observed_positions", "apex_observed_tail_mass_L_ge_n",
    ]
    cols = [c for c in extra + wanted if c in numeric_cols]
    agg = df.groupby(group_cols, dropna=False)[cols].mean().reset_index()
    counts = df.groupby(group_cols, dropna=False).size().rename("n").reset_index()
    return counts.merge(agg, on=group_cols, how="left")


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Correlations over request-level rows, not over exploded positions."""
    df = survival_subset(df).reset_index(drop=True)
    targets = [c for c in ["speedup_vs_ar", "tokens_per_sec"] if c in df.columns]
    predictors = [c for c in [
        "acceptance_rate",
        "accepted_per_draft",
        "apex_expected_len_survival_observed",
        "apex_expected_len_counter",
        "mean_entropy",
        "repetition_density",
        "apex_tail_expected_missing",
    ] if c in df.columns]
    rows = []
    for target in targets:
        for pred in predictors:
            sub = df[[target, pred]].replace([np.inf, -np.inf], np.nan).dropna()
            # Important invariant: n must never exceed request-level survival rows.
            if len(sub) > len(df):
                raise RuntimeError(f"correlation bug: n={len(sub)} > request_rows={len(df)} for {target}/{pred}")
            if len(sub) < 3 or sub[target].nunique() <= 1 or sub[pred].nunique() <= 1:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(sub[target].corr(sub[pred], method="pearson"))
                spearman = float(sub[target].corr(sub[pred], method="spearman"))
            rows.append({"target": target, "predictor": pred, "n": int(len(sub)), "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(rows)


def _scatter(df: pd.DataFrame, x: str, y: str, out: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    cols = [x, y] + (["method"] if "method" in df.columns else [])
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(7.2, 5.0))
    if "method" in sub.columns:
        for method, part in sub.groupby("method"):
            plt.scatter(part[x], part[y], label=str(method), alpha=0.55, s=22)
        plt.legend(fontsize=8)
    else:
        plt.scatter(sub[x], sub[y], alpha=0.55, s=22)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_acceptance_vs_speedup(df: pd.DataFrame, fig_dir: Path) -> None:
    df = survival_subset(df)
    if "acceptance_rate" in df.columns and "speedup_vs_ar" in df.columns:
        _scatter(df, "acceptance_rate", "speedup_vs_ar", fig_dir / "layer1_acceptance_rate_vs_speedup.png",
                 title="Average token acceptance rate vs speedup", xlabel="average token acceptance rate", ylabel="speedup vs AR")


def plot_expected_len_vs_speedup(df: pd.DataFrame, fig_dir: Path) -> None:
    df = survival_subset(df)
    if "apex_expected_len_counter" in df.columns and "speedup_vs_ar" in df.columns:
        _scatter(df, "apex_expected_len_counter", "speedup_vs_ar", fig_dir / "layer1_counter_expected_len_vs_speedup.png",
                 title="Counter expected accepted length vs speedup", xlabel="counter E[L] = accepted tokens / draft", ylabel="speedup vs AR")
    if "apex_expected_len_survival_observed" in df.columns and "speedup_vs_ar" in df.columns:
        _scatter(df, "apex_expected_len_survival_observed", "speedup_vs_ar", fig_dir / "layer1_observed_survival_expected_len_vs_speedup.png",
                 title="Observed survival-prefix E[L] vs speedup", xlabel="observed survival sum (prefix only if truncated)", ylabel="speedup vs AR")


def plot_survival_curves(grouped: pd.DataFrame, fig_dir: Path, *, max_depth: int | None) -> None:
    survival_cols = sorted([c for c in grouped.columns if c.startswith("survival_pos_")], key=lambda c: int(c.rsplit("_", 1)[1]))
    if max_depth is not None:
        survival_cols = [c for c in survival_cols if int(c.rsplit("_", 1)[1]) < max_depth]
    if not survival_cols or grouped.empty:
        return
    # One curve per method/depth averaged across workloads to avoid a spaghetti plot.
    label_cols = [c for c in ["method", "k"] if c in grouped.columns]
    avg = grouped.groupby(label_cols, dropna=False)[survival_cols].mean().reset_index() if label_cols else grouped
    plt.figure(figsize=(8.0, 5.2))
    for _, row in avg.iterrows():
        y = [row.get(c, np.nan) for c in survival_cols]
        if all(pd.isna(v) for v in y):
            continue
        label = "/".join(str(row[c]) for c in label_cols) if label_cols else "survival"
        plt.plot(range(len(y)), y, marker="o", linewidth=1.8, alpha=0.85, label=label)
    plt.title("Accepted-length survival curves by method/depth")
    plt.xlabel("draft position j")
    plt.ylabel(r"survival $S_j=P(L>j)$")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "layer1_survival_curves_by_method_depth.png", dpi=180)
    plt.close()


def plot_survival_curves_by_workload(grouped: pd.DataFrame, fig_dir: Path, *, max_depth: int | None) -> None:
    survival_cols = sorted([c for c in grouped.columns if c.startswith("survival_pos_")], key=lambda c: int(c.rsplit("_", 1)[1]))
    if max_depth is not None:
        survival_cols = [c for c in survival_cols if int(c.rsplit("_", 1)[1]) < max_depth]
    if not survival_cols or grouped.empty or "workload" not in grouped.columns:
        return
    workloads = list(grouped["workload"].dropna().unique())[:8]
    for workload in workloads:
        part = grouped[grouped["workload"] == workload]
        label_cols = [c for c in ["method", "k"] if c in part.columns]
        avg = part.groupby(label_cols, dropna=False)[survival_cols].mean().reset_index() if label_cols else part
        plt.figure(figsize=(8.0, 5.2))
        for _, row in avg.iterrows():
            y = [row.get(c, np.nan) for c in survival_cols]
            if all(pd.isna(v) for v in y):
                continue
            label = "/".join(str(row[c]) for c in label_cols) if label_cols else "survival"
            plt.plot(range(len(y)), y, marker="o", linewidth=1.5, alpha=0.85, label=label)
        plt.title(f"Survival curves: {workload}")
        plt.xlabel("draft position j")
        plt.ylabel(r"survival $S_j=P(L>j)$")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        safe = str(workload).replace("/", "_").replace(" ", "_")
        plt.savefig(fig_dir / f"layer1_survival_curves_{safe}.png", dpi=180)
        plt.close()


def plot_expected_len_vs_oracle(df: pd.DataFrame, fig_dir: Path) -> None:
    df = survival_subset(df)
    y = "oracle_k_estimated"
    if "apex_expected_len_counter" in df.columns and y in df.columns:
        _scatter(df, "apex_expected_len_counter", y, fig_dir / "layer1_expected_len_vs_oracle_k.png",
                 title="Expected accepted length vs oracle depth", xlabel="counter E[L]", ylabel="oracle estimated k")


def plot_draft_model_counterexample(df: pd.DataFrame, fig_dir: Path) -> None:
    df = survival_subset(df)
    needed = ["method", "acceptance_rate", "speedup_vs_ar", "apex_expected_len_counter"]
    if not all(c in df.columns for c in needed):
        return
    sub = df[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return
    means = sub.groupby("method", dropna=False).mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for _, row in means.iterrows():
        ax[0].scatter(row["acceptance_rate"], row["speedup_vs_ar"], s=95)
        ax[0].text(row["acceptance_rate"], row["speedup_vs_ar"], " " + str(row["method"]), fontsize=9)
        ax[1].scatter(row["apex_expected_len_counter"], row["speedup_vs_ar"], s=95)
        ax[1].text(row["apex_expected_len_counter"], row["speedup_vs_ar"], " " + str(row["method"]), fontsize=9)
    for a in ax:
        a.axhline(1.0, linestyle="--", linewidth=1.0)
        a.grid(True, alpha=0.25)
    ax[0].set_title("Acceptance alone can mislead")
    ax[0].set_xlabel("mean token acceptance rate")
    ax[0].set_ylabel("mean speedup vs AR")
    ax[1].set_title("Cost still matters even when E[L] is high")
    ax[1].set_xlabel("mean counter E[L]")
    ax[1].set_ylabel("mean speedup vs AR")
    plt.tight_layout()
    plt.savefig(fig_dir / "layer1_draft_model_counterexample.png", dpi=180)
    plt.close()


def write_diagnostics(request_df: pd.DataFrame, grouped_df: pd.DataFrame, out_dir: Path) -> None:
    lines = []
    n_total = len(request_df)
    n_surv = len(survival_subset(request_df))
    lines.append(f"request_rows_total: {n_total}")
    lines.append(f"request_rows_with_survival_non_ar: {n_surv}")
    if "method" in request_df.columns:
        lines.append("methods_total:")
        lines.extend(f"  {k}: {v}" for k, v in request_df["method"].value_counts(dropna=False).items())
    if "apex_observed_positions" in request_df.columns:
        lines.append("observed_positions_non_ar:")
        lines.extend(f"  {k}: {v}" for k, v in survival_subset(request_df)["apex_observed_positions"].value_counts(dropna=False).sort_index().items())
    if "apex_tail_expected_missing" in request_df.columns:
        tail = survival_subset(request_df)["apex_tail_expected_missing"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tail):
            lines.append(f"tail_missing_mean: {tail.mean():.6f}")
            lines.append(f"tail_missing_p95: {tail.quantile(0.95):.6f}")
    (out_dir / "layer1_diagnostics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(request_df: pd.DataFrame, grouped_df: pd.DataFrame, fig_dir: Path, *, max_depth: int | None) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_acceptance_vs_speedup(request_df, fig_dir)
    plot_expected_len_vs_speedup(request_df, fig_dir)
    plot_survival_curves(grouped_df, fig_dir, max_depth=max_depth)
    plot_survival_curves_by_workload(grouped_df, fig_dir, max_depth=max_depth)
    plot_expected_len_vs_oracle(request_df, fig_dir)
    plot_draft_model_counterexample(request_df, fig_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs", required=True, help="Path to results/all_runs.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for Layer 1 CSVs/figures")
    parser.add_argument("--max-depth", type=int, default=16, help="Maximum draft positions to analyze")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.all_runs)
    request_df = enrich_dataframe(df, max_depth=args.max_depth)
    grouped_df = group_survival(request_df)
    corr_df = correlation_table(request_df)

    request_df.to_csv(out_dir / "layer1_request_survival.csv", index=False)
    grouped_df.to_csv(out_dir / "layer1_group_survival.csv", index=False)
    corr_df.to_csv(out_dir / "layer1_correlations.csv", index=False)
    write_diagnostics(request_df, grouped_df, out_dir)
    make_figures(request_df, grouped_df, out_dir / "figures", max_depth=args.max_depth)

    print(f"[Layer1] wrote {out_dir / 'layer1_request_survival.csv'} ({len(request_df)} rows)")
    print(f"[Layer1] wrote {out_dir / 'layer1_group_survival.csv'} ({len(grouped_df)} rows)")
    print(f"[Layer1] wrote {out_dir / 'layer1_correlations.csv'}")
    print(f"[Layer1] wrote {out_dir / 'layer1_diagnostics.txt'}")
    print(f"[Layer1] figures -> {out_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
