#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.apex_eval_utils import (
    add_waste_percent,
    concat_with_run_metadata,
    ensure_dir,
    run_cmd,
    save_bar_plot,
    save_grouped_line_plot,
    write_json,
)


SUMMARY_FILES = [
    "request_summary_overall.csv",
    "request_summary_by_workload.csv",
    "block_summary_overall.csv",
    "block_summary_by_workload.csv",
    "active_k_distribution.csv",
]


def summarize_one_run(root: Path, force: bool = False) -> bool:
    summary_dir = root / "summary"
    need = force or not (summary_dir / "request_summary_overall.csv").exists()
    if not need:
        return True
    script = Path("src/apexp/week2/summarize_learned_hot_run.py")
    if not script.exists():
        print(f"[warn] missing {script}; cannot summarize {root}")
        return False
    try:
        run_cmd(["python3", str(script), "--root", str(root), "--out-dir", str(summary_dir)], check=True)
        return True
    except Exception as e:
        print(f"[warn] failed summarizing {root}: {e}")
        return False


def find_runs(runs_root: Path) -> list[Path]:
    out = []
    for p in sorted(runs_root.iterdir() if runs_root.exists() else []):
        if not p.is_dir():
            continue
        # A hot run has a results directory or existing summary.
        if (p / "results").exists() or (p / "summary").exists() or (p / "run.log").exists():
            out.append(p)
    return out


def add_ratios(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or "pool" not in df.columns or "mean_tps" not in df.columns:
        return df
    out = df.copy()
    base_cols = group_cols + ["mean_tps"]
    base = out[out["pool"] == "slow_only"][base_cols].rename(columns={"mean_tps": "slow_only_mean_tps"})
    out = out.merge(base, on=group_cols, how="left")
    out["tps_ratio_vs_slow_only"] = pd.to_numeric(out["mean_tps"], errors="coerce") / pd.to_numeric(out["slow_only_mean_tps"], errors="coerce")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = ensure_dir(args.out_dir)
    plot_dir = ensure_dir(out_dir / "plots")

    runs = find_runs(runs_root)
    print(f"found {len(runs)} runs under {runs_root}")

    ok_runs = []
    for r in runs:
        if summarize_one_run(r, force=args.force):
            ok_runs.append(r)

    tables = {}
    for fname in SUMMARY_FILES:
        df = concat_with_run_metadata(ok_runs, fname)
        if not df.empty:
            df = add_waste_percent(df)
        tables[fname] = df
        if not df.empty:
            df.to_csv(out_dir / fname.replace(".csv", "_all_runs.csv"), index=False)

    req_overall = tables.get("request_summary_overall.csv", pd.DataFrame())
    blk_overall = tables.get("block_summary_overall.csv", pd.DataFrame())
    req_w = tables.get("request_summary_by_workload.csv", pd.DataFrame())
    blk_w = tables.get("block_summary_by_workload.csv", pd.DataFrame())
    active = tables.get("active_k_distribution.csv", pd.DataFrame())

    # Candidate-k and temperature ablations use slow_fast rows.
    if not req_overall.empty:
        req_overall = add_ratios(req_overall, ["run_name"])
        req_overall.to_csv(out_dir / "request_overall_all_runs.csv", index=False)
        sf = req_overall[req_overall["pool"] == "slow_fast"].copy()
        sf.to_csv(out_dir / "candidate_k_ablation.csv", index=False)
        sf.to_csv(out_dir / "temperature_ablation.csv", index=False)
        save_bar_plot(sf.sort_values("mean_tps", ascending=False), "run_name", "mean_tps", plot_dir / "slow_fast_mean_tps_by_run.png", "APEX slow_fast mean TPS by run")
        save_bar_plot(sf.sort_values("tps_ratio_vs_slow_only", ascending=False), "run_name", "tps_ratio_vs_slow_only", plot_dir / "slow_fast_tps_ratio_by_run.png", "APEX slow_fast TPS ratio vs slow_only")

    if not blk_overall.empty:
        blk_overall = add_waste_percent(blk_overall)
        blk_overall.to_csv(out_dir / "block_overall_all_runs.csv", index=False)
        sf_b = blk_overall[blk_overall["pool"] == "slow_fast"].copy()
        save_bar_plot(sf_b.sort_values("wasted_token_rate"), "run_name", "wasted_token_pct", plot_dir / "slow_fast_waste_pct_by_run.png", "APEX slow_fast wasted token %")
        if "STE_vs_slow_only" in sf_b.columns:
            save_bar_plot(sf_b.sort_values("STE_vs_slow_only", ascending=False), "run_name", "STE_vs_slow_only", plot_dir / "slow_fast_STE_by_run.png", "APEX slow_fast STE vs slow_only")
        if "ATE_vs_slow_only" in sf_b.columns:
            save_bar_plot(sf_b.sort_values("ATE_vs_slow_only", ascending=False), "run_name", "ATE_vs_slow_only", plot_dir / "slow_fast_ATE_by_run.png", "APEX slow_fast ATE vs slow_only")

    if not req_w.empty:
        req_w.to_csv(out_dir / "request_by_workload_all_runs.csv", index=False)
        sfw = req_w[req_w["pool"] == "slow_fast"].copy()
        if not sfw.empty:
            save_grouped_line_plot(sfw, "workload", "mean_tps", "run_name", plot_dir / "mean_tps_by_workload.png", "Mean TPS by workload")

    if not blk_w.empty:
        blk_w = add_waste_percent(blk_w)
        blk_w.to_csv(out_dir / "block_by_workload_all_runs.csv", index=False)
        sfw = blk_w[blk_w["pool"] == "slow_fast"].copy()
        if not sfw.empty:
            save_grouped_line_plot(sfw, "workload", "wasted_token_pct", "run_name", plot_dir / "waste_pct_by_workload.png", "Wasted token % by workload")

    if not active.empty:
        active.to_csv(out_dir / "active_k_distribution_all_runs.csv", index=False)
        write_json(out_dir / "run_roots.json", [str(r) for r in ok_runs])

    print(f"WROTE SUMMARY TO: {out_dir}")


if __name__ == "__main__":
    main()
