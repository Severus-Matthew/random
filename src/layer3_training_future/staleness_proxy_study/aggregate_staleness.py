#!/usr/bin/env python
"""
aggregate_staleness.py — turn staleness runs into the Phase-3 decision curves
=============================================================================
Reads every summary.csv under experiments named
  S_stale__<study>__<family>__<base_tag>
and joins each to its drift metadata via the authoritative index written by
run_staleness_study.py (results/staleness_index.csv) — so family / drift
distance attribution never depends on manifest tag-prefix consistency.

Outputs:
  staleness_cells.csv            per (study, family, drift, method, k, temp, workload)
  staleness_retention.csv        acceptance(drift) / acceptance(fresh)
  staleness_slopes.csv           per (study, family, method): slope of acceptance
                                 vs drift distance  (the staleness-sensitivity #)
  staleness_position_decay.csv   per-position accept rate vs drift

Usage:
  python aggregate_staleness.py \
      --runs_dir results/Runs/Phase2 \
      --index results/staleness_index.csv \
      --kl 'results/drifted_*/drift_kl.csv' \
      --out_dir results/staleness
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXP_RE = re.compile(r"^S_stale__(?P<study>.+?)__(?P<family>[^_]+(?:_[^_]+)*?)__(?P<base_tag>.+)$")
POS_COLS = ["accept_rate_pos_0", "accept_rate_pos_1",
            "accept_rate_pos_2", "accept_rate_pos_3"]


def load_runs(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for p in runs_dir.glob("**/summary.csv"):
        if "S_stale__" not in str(p):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "experiment" not in df.columns or df.empty:
            continue
        frames.append(df)
    if not frames:
        raise SystemExit(f"No S_stale__* summaries under {runs_dir}. "
                         "Run run_staleness_study.py first.")
    return pd.concat(frames, ignore_index=True)


def load_kl(kl_glob) -> pd.DataFrame | None:
    if not kl_glob:
        return None
    files = glob.glob(kl_glob)
    if not files:
        return None
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f))
        except Exception:
            pass
    if not parts:
        return None
    kl = pd.concat(parts, ignore_index=True)
    return kl if "mean_token_kl" in kl.columns else None


def cell_table(df, index, kl):
    metrics = {
        "n": ("id", "count"),
        "acceptance_rate": ("acceptance_rate", "mean"),
        "accepted_per_draft": ("accepted_per_draft", "mean"),
        "tps_mean": ("tokens_per_sec", "mean"),
        "rejection_rate": ("rejection_rate", "mean"),
        "first_rejection_pos": ("first_rejection_pos", "mean"),
        "rejection_concentration": ("rejection_concentration", "mean"),
        "mean_entropy": ("mean_entropy", "mean"),
    }
    for c in POS_COLS:
        if c in df.columns:
            metrics[c] = (c, "mean")
    avail = {k: v for k, v in metrics.items() if v[0] in df.columns}
    cells = (df.groupby(["experiment", "method", "k", "temperature", "workload"],
                        dropna=False)
               .agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1])
                       for k, v in avail.items()})
               .reset_index())

    # authoritative join on experiment -> study/family/drift/distance
    idx_cols = ["experiment", "study", "family", "drift_tag", "base_tag",
                "mode", "param", "rel_weight_dist"]
    cells = cells.merge(index[[c for c in idx_cols if c in index.columns]],
                        on="experiment", how="left")

    # optional output-KL (join on the original manifest tag)
    if kl is not None:
        k = kl.rename(columns={"tag": "drift_tag"})[["drift_tag", "mean_token_kl"]]
        cells = cells.merge(k, on="drift_tag", how="left")

    cells["param"] = pd.to_numeric(cells.get("param"), errors="coerce")
    cells["rel_weight_dist"] = pd.to_numeric(cells.get("rel_weight_dist"), errors="coerce")
    return cells


def is_fresh(row):
    p = row.get("param")
    if pd.isna(p):
        return False
    if row.get("mode") == "interp":
        return abs(float(p) - 1.0) < 1e-9
    if row.get("mode") == "noise":
        return abs(float(p)) < 1e-12
    return False


def add_retention(cells):
    cells = cells.copy()
    cells["is_fresh"] = cells.apply(is_fresh, axis=1)
    grp = ["study", "family", "method", "k", "temperature", "workload", "mode"]
    fresh = (cells[cells["is_fresh"]].groupby(grp)["acceptance_rate"]
             .mean().rename("acceptance_fresh").reset_index())
    cells = cells.merge(fresh, on=grp, how="left")
    cells["acceptance_retention"] = (cells["acceptance_rate"]
                                     / cells["acceptance_fresh"].replace(0, np.nan))
    return cells


def slopes(cells, x_col):
    rows = []
    for (study, family, method, mode), g in cells.groupby(
            ["study", "family", "method", "mode"]):
        g = g.dropna(subset=[x_col, "acceptance_rate"])
        if g[x_col].nunique() < 2:
            continue
        x = g[x_col].to_numpy(float); y = g["acceptance_rate"].to_numpy(float)
        A = np.vstack([x, np.ones_like(x)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ np.array([slope, intercept])
        ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(dict(study=study, family=family, method=method, mode=mode,
                         x=x_col, slope=round(slope, 5), intercept=round(intercept, 5),
                         r2=round(r2, 4), n_points=len(g)))
    return pd.DataFrame(rows)


def position_decay(cells):
    have = [c for c in POS_COLS if c in cells.columns]
    if not have:
        return pd.DataFrame()
    keys = ["study", "family", "method", "k", "temperature", "mode", "param",
            "rel_weight_dist"]
    keys = [c for c in keys if c in cells.columns]
    long = cells.melt(id_vars=keys, value_vars=have,
                      var_name="position", value_name="accept_rate")
    long["position"] = long["position"].str.extract(r"(\d+)").astype(int)
    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/Runs/Phase2")
    ap.add_argument("--index", required=True, help="results/staleness_index.csv")
    ap.add_argument("--kl", default=None,
                    help="glob for per-family drift_kl.csv, e.g. 'results/drifted_*/drift_kl.csv'")
    ap.add_argument("--out_dir", default="results/staleness")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_runs(Path(args.runs_dir))
    index = pd.read_csv(args.index)
    kl = load_kl(args.kl)

    cells = cell_table(df, index, kl)
    cells = add_retention(cells)
    cells.to_csv(out / "staleness_cells.csv", index=False)
    print(f"  -> staleness_cells.csv ({len(cells)} cells)")

    cells[["study", "family", "method", "k", "temperature", "workload", "mode",
           "param", "rel_weight_dist", "acceptance_rate", "acceptance_fresh",
           "acceptance_retention"]].to_csv(out / "staleness_retention.csv", index=False)
    print("  -> staleness_retention.csv")

    sl = slopes(cells, "rel_weight_dist")
    if kl is not None and "mean_token_kl" in cells.columns:
        sl = pd.concat([sl, slopes(cells, "mean_token_kl")], ignore_index=True)
    sl.to_csv(out / "staleness_slopes.csv", index=False)
    print("  -> staleness_slopes.csv")

    pdc = position_decay(cells)
    if not pdc.empty:
        pdc.to_csv(out / "staleness_position_decay.csv", index=False)
        print("  -> staleness_position_decay.csv")

    head = sl[(sl["study"] == "S1_core_drift_curve") & (sl["mode"] == "interp")
              & (sl["x"] == "rel_weight_dist")]
    if not head.empty:
        print("\n── Staleness sensitivity (S1, interp): acceptance vs weight-drift ──")
        print("   (more negative slope = more fragile; ~0 = staleness-robust)")
        print(head[["family", "method", "slope", "r2", "n_points"]]
              .sort_values(["family", "slope"]).to_string(index=False))


if __name__ == "__main__":
    main()
