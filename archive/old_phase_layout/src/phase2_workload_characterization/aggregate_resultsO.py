#!/usr/bin/env python
"""
aggregate_results.py — Phase 1 + Phase 2 aggregation
=====================================================
Reads all summary.csv files under results_dir and produces:

  all_runs.csv                          flat concat of every row
  aggregate_by_workload_method.csv      workload × method × k
  aggregate_k_sweep.csv                 k sensitivity (exp A + B)
  aggregate_temperature_sweep.csv       temperature effect (exp E)
  aggregate_context_growth.csv          context length scaling (exp F)
  aggregate_draft_model_sweep.csv       Qwen3 draft family (exp D)
  aggregate_ngram_config_sweep.csv      ngram window configs (exp C)
  aggregate_multi_turn.csv              multi-turn depth (exp G)
  aggregate_positional_acceptance.csv   per-position accept rates (eagle3/draft)
  grand_summary_table.csv               human-readable summary (k=4 baseline)
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# ── All metrics to aggregate ──────────────────────────────────────────────
CORE_AGG = {
    # bookkeeping
    "n":                      ("id",                                "count"),
    # throughput
    "tps_mean":               ("tokens_per_sec",                    "mean"),
    "tps_p50":                ("tokens_per_sec",                    "median"),
    "tps_std":                ("tokens_per_sec",                    "std"),
    "tps_p95":                ("tokens_per_sec",                    lambda x: x.quantile(0.95)),
    # latency
    "lat_mean":               ("latency_s",                         "mean"),
    "lat_p50":                ("latency_s",                         "median"),
    "lat_p95":                ("latency_s",                         lambda x: x.quantile(0.95)),
    "lat_std":                ("latency_s",                         "std"),
    # output tokens
    "n_tokens_mean":          ("n_output_tokens",                   "mean"),
    # acceptance (real from vLLM counters)
    "accept_rate_mean":       ("acceptance_rate",                   "mean"),
    "accept_rate_std":        ("acceptance_rate",                   "std"),
    "accept_per_vpass_mean":  ("accepted_tokens_per_verifier_pass", "mean"),
    "draft_tokens_mean":      ("draft_tokens",                      "mean"),
    "accepted_tokens_mean":   ("accepted_tokens_total",             "mean"),
    # per-position acceptance profile (eagle3/draft_sd)
    "accept_pos0_mean":       ("accept_rate_pos_0",                 "mean"),
    "accept_pos1_mean":       ("accept_rate_pos_1",                 "mean"),
    "accept_pos2_mean":       ("accept_rate_pos_2",                 "mean"),
    "accept_pos3_mean":       ("accept_rate_pos_3",                 "mean"),
    "accepted_per_draft_mean":("accepted_per_draft",                "mean"),
    # phase 1 required
    "rollback_freq_mean":     ("rollback_frequency",                "mean"),
    "rollback_freq_std":      ("rollback_frequency",                "std"),
    "verif_util_mean":        ("verifier_utilization",              "mean"),
    # structural / token regime
    "entropy_mean":           ("mean_entropy",                      "mean"),
    "entropy_std":            ("mean_entropy",                      "std"),
    "rep_density_mean":       ("repetition_density",                "mean"),
    "acc_vol_mean":           ("acceptance_volatility",             "mean"),
    # engine latency breakdown (where available)
    "ttft_mean":              ("ttft_s",                            "mean"),
    "itl_mean":               ("itl_s",                            "mean"),
    "prefill_mean":           ("prefill_time_s",                    "mean"),
    "decode_mean":            ("decode_time_s",                     "mean"),
    "tpot_mean":              ("tpot_s",                            "mean"),
    # cache metrics
    "prefix_cache_hit_rate":  ("prefix_cache_hit_rate",             "mean"),
    "prompt_cache_frac":      ("prompt_cache_frac",                 "mean"),
    "kv_cache_usage":         ("kv_cache_usage_perc",               "mean"),
    # ── rejection metrics (exp K) ──────────────────────────────────────────
    "rejected_tokens_mean":      ("rejected_tokens",          "mean"),
    "rejection_rate_mean":       ("rejection_rate",           "mean"),
    "rejection_rate_std":        ("rejection_rate",           "std"),
    "first_rejection_pos_mean":  ("first_rejection_pos",      "mean"),
    "rejection_conc_mean":       ("rejection_concentration",  "mean"),
    "rejection_severity_mean":   ("rejection_severity",       "mean"),
    "decay_slope_mean":          ("acceptance_decay_slope",   "mean"),
    # ── intra-sequence window metrics (exp J) ──────────────────────────────
    "entropy_w0_mean":           ("entropy_w0_mean",          "mean"),
    "entropy_w1_mean":           ("entropy_w1_mean",          "mean"),
    "entropy_w2_mean":           ("entropy_w2_mean",          "mean"),
    "entropy_trend_mean":        ("entropy_trend",            "mean"),
    "entropy_range_mean":        ("entropy_range",            "mean"),
    "high_entropy_frac_mean":    ("high_entropy_frac",        "mean"),
    # ── oracle k + entropy bucket (exp H, I) ──────────────────────────────
    "oracle_k_mean":             ("oracle_k_estimated",       "mean"),
    "oracle_gain_mean":          ("oracle_k_vs_k4_gain",      "mean"),
    "oracle_expected_mean":      ("oracle_expected_tokens",   "mean"),
}


def safe_agg(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Aggregate only columns that exist in df."""
    existing = {k: v for k, v in CORE_AGG.items() if v[0] in df.columns}
    if not existing:
        return df.groupby(group_cols, dropna=False).size().reset_index(name="n")
    result = (df.groupby(group_cols, dropna=False)
                .agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1])
                        for k, v in existing.items()})
                .reset_index())
    return result


def load_all(results_dir: Path) -> pd.DataFrame:
    """Recursively load all summary.csv files, enriching with path metadata."""
    paths = sorted(results_dir.glob("**/summary.csv"))
    if not paths:
        raise SystemExit(f"No summary.csv found under {results_dir}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        parts = p.parts

        # Path fallback for older runs that may not have all columns
        if "experiment" not in df.columns:
            df["experiment"] = parts[-5] if len(parts) >= 5 else "unknown"
        if "workload" not in df.columns:
            df["workload"] = parts[-4] if len(parts) >= 4 else "unknown"
        if "method" not in df.columns:
            df["method"] = parts[-3] if len(parts) >= 3 else "unknown"
        if "k" not in df.columns and len(parts) >= 2:
            k_str = str(parts[-2]).replace("k", "").split("_")[0]
            try:
                df["k"] = int(k_str)
            except ValueError:
                df["k"] = np.nan
        # Phase 2 columns — fill with defaults if missing (older Phase 1 runs)
        for col, default in [
            ("temperature", 0.0),
            ("max_prompt_tokens", 0),
            ("num_turns", 1),
            ("ngram_lookup_min", 1),
            ("ngram_lookup_max", 4),
            ("draft_model", ""),
            ("eagle3_model", ""),
            ("accept_rate_pos_0", np.nan),
            ("accept_rate_pos_1", np.nan),
            ("accept_rate_pos_2", np.nan),
            ("accept_rate_pos_3", np.nan),
            ("accepted_per_draft", np.nan),
            ("draft_tokens", np.nan),
            ("accepted_tokens_total", np.nan),
            ("ttft_s", np.nan),
            ("itl_s", np.nan),
            ("prefill_time_s", np.nan),
            ("decode_time_s", np.nan),
            ("tpot_s", np.nan),
            ("prefix_cache_hit_rate", np.nan),
            ("prompt_cache_frac", np.nan),
            ("kv_cache_usage_perc", np.nan),
        ]:
            if col not in df.columns:
                df[col] = default

        df["_src"] = str(p)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(combined)} rows from {len(paths)} files")
    return combined


def add_speedup_vs_ar(df: pd.DataFrame) -> pd.DataFrame:
    """Merge AR TPS baseline and compute speedup ratio."""
    if "tps_mean" not in df.columns:
        return df
    ar = (df[df["method"] == "ar"]
            .groupby("workload")["tps_mean"].mean()
            .rename("ar_tps")
            .reset_index())
    df = df.merge(ar, on="workload", how="left")
    df["speedup_vs_ar"] = df["tps_mean"] / df["ar_tps"].replace(0, np.nan)
    return df.drop(columns=["ar_tps"], errors="ignore")


def write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  -> {path.name}  ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results",
                    help="Root directory containing summary.csv files")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: same as results_dir)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {results_dir} ...")
    df = load_all(results_dir)
    df.drop(columns=["_src"], errors="ignore", inplace=True)

    # ── 1. Flat dump ──────────────────────────────────────────────────────
    write(df, out_dir / "all_runs.csv")

    # ── 2. Baseline: workload × method × k (all experiments) ─────────────
    agg = safe_agg(df, ["experiment", "workload", "method", "k"])
    agg = add_speedup_vs_ar(agg)
    write(agg, out_dir / "aggregate_by_workload_method.csv")

    # ── 3. k sweep (exp A + B) ────────────────────────────────────────────
    ksweep = df[
        df["experiment"].str.contains("A_baseline|B_k_sweep", na=False, regex=True)
    ].copy()
    if not ksweep.empty:
        out = safe_agg(ksweep, ["workload", "method", "k"])
        out = add_speedup_vs_ar(out)
        write(out, out_dir / "aggregate_k_sweep.csv")

    # ── 4. Temperature sweep (exp E + A baseline at T=0) ─────────────────
    tdf = df[
        df["experiment"].str.contains("E_temperature|A_baseline", na=False, regex=True)
    ].copy()
    if not tdf.empty and "temperature" in tdf.columns:
        out = safe_agg(tdf, ["workload", "method", "k", "temperature"])
        out = add_speedup_vs_ar(out)
        write(out, out_dir / "aggregate_temperature_sweep.csv")

    # ── 5. Context growth (exp F) ─────────────────────────────────────────
    cdf = df[df["experiment"].str.startswith("F_context", na=False)].copy()
    if not cdf.empty and "max_prompt_tokens" in cdf.columns:
        out = safe_agg(cdf, ["workload", "method", "k", "max_prompt_tokens"])
        out = add_speedup_vs_ar(out)
        write(out, out_dir / "aggregate_context_growth.csv")

    # ── 6. Draft model sweep (exp D) ─────────────────────────────────────
    ddf = df[df["experiment"].str.startswith("D_draft", na=False)].copy()
    if not ddf.empty and "draft_model" in ddf.columns:
        out = safe_agg(ddf, ["workload", "method", "k", "draft_model"])
        out = add_speedup_vs_ar(out)
        write(out, out_dir / "aggregate_draft_model_sweep.csv")

    # ── 7. Ngram config sweep (exp C) ─────────────────────────────────────
    ndf = df[df["experiment"].str.startswith("C_ngram", na=False)].copy()
    if not ndf.empty:
        gcols = ["workload", "method", "k"]
        if "ngram_lookup_min" in ndf.columns: gcols += ["ngram_lookup_min"]
        if "ngram_lookup_max" in ndf.columns: gcols += ["ngram_lookup_max"]
        out = safe_agg(ndf, gcols)
        write(out, out_dir / "aggregate_ngram_config_sweep.csv")

    # ── 8. Multi-turn (exp G) ─────────────────────────────────────────────
    mdf = df[df["experiment"].str.startswith("G_multi", na=False)].copy()
    if not mdf.empty and "num_turns" in mdf.columns:
        out = safe_agg(mdf, ["workload", "method", "k", "num_turns"])
        out = add_speedup_vs_ar(out)
        write(out, out_dir / "aggregate_multi_turn.csv")

    # ── 9. Per-position acceptance profile (Phase 2 novel metric) ─────────
    pos_cols = [c for c in df.columns if c.startswith("accept_rate_pos_")]
    spec_df  = df[df["method"].isin(["eagle3", "draft_sd", "ngram_sd"])].copy()
    if pos_cols and not spec_df.empty:
        gcols = ["experiment", "workload", "method", "k"]
        if "temperature" in spec_df.columns: gcols += ["temperature"]
        pos_agg = spec_df.groupby(gcols, dropna=False)[pos_cols].mean().reset_index()
        # Melt into long format for easier plotting
        pos_long = pos_agg.melt(
            id_vars=gcols,
            value_vars=pos_cols,
            var_name="position",
            value_name="accept_rate_at_position",
        )
        pos_long["position"] = pos_long["position"].str.extract(r"(\d+)").astype(float)
        write(pos_long, out_dir / "aggregate_positional_acceptance.csv")

    # ── 10. Oracle k study (exp H) ──────────────────────────────────────────
    hdf = df[df["experiment"].str.startswith("H_oracle", na=False)].copy()
    if not hdf.empty:
        out = safe_agg(hdf, ["workload", "method", "k"])
        out = add_speedup_vs_ar(out)
        write(out, out_dir / "aggregate_oracle_k.csv")
        # Per-prompt oracle k distribution (from oracle_k_estimated column)
        if "oracle_k_estimated" in hdf.columns and "entropy_bucket" in hdf.columns:
            oracle_detail = (hdf[hdf["method"].isin(["eagle3","ngram_sd"])]
                             .groupby(["workload","method","k","entropy_bucket"])
                             .agg(oracle_k_mean     =("oracle_k_estimated",  "mean"),
                                  oracle_gain_mean  =("oracle_k_vs_k4_gain", "mean"),
                                  oracle_gain_pct   =("oracle_k_vs_k4_gain",
                                                       lambda x: (x > 0).mean() * 100),
                                  n                 =("oracle_k_estimated",  "count"))
                             .reset_index())
            write(oracle_detail, out_dir / "aggregate_oracle_k_by_bucket.csv")

    # ── 11. Entropy bucket study (exp I) ─────────────────────────────────
    idf = df[df["experiment"].str.startswith("I_entropy", na=False)].copy()
    if not idf.empty and "entropy_bucket" in idf.columns:
        # Speedup by bucket × method × k
        out = safe_agg(idf, ["workload", "method", "k", "entropy_bucket"])
        out = add_speedup_vs_ar(out)
        write(out, out_dir / "aggregate_entropy_bucket.csv")
        # Bucket summary: best method and k per bucket
        if "tps_mean" in out.columns and "speedup_vs_ar" in out.columns:
            best_per_bucket = (out[out["method"] != "ar"]
                               .sort_values("speedup_vs_ar", ascending=False)
                               .groupby(["workload", "entropy_bucket"])
                               .first()
                               .reset_index()[["workload","entropy_bucket",
                                               "method","k","tps_mean","speedup_vs_ar"]])
            write(best_per_bucket, out_dir / "aggregate_entropy_bucket_best.csv")

    # ── 12. Intra-sequence transition (exp J) ────────────────────────────
    jdf = df[df["experiment"].str.startswith("J_intra", na=False)].copy()
    if not jdf.empty:
        win_cols = [c for c in jdf.columns if c.startswith("entropy_w")]
        extra    = [c for c in ["entropy_trend","entropy_range","high_entropy_frac"]
                    if c in jdf.columns]
        if win_cols or extra:
            gcols = ["workload", "method", "k"]
            agg_dict = {c: (c, "mean") for c in win_cols + extra if c in jdf.columns}
            if agg_dict:
                out = (jdf.groupby(gcols, dropna=False)
                          .agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1])
                                  for k, v in agg_dict.items()})
                          .reset_index())
                write(out, out_dir / "aggregate_intra_sequence.csv")

    # ── 13. Rejection study (exp K) ──────────────────────────────────────
    kdf = df[df["experiment"].str.startswith("K_rejection", na=False)].copy()
    if not kdf.empty:
        rej_cols = ["rejected_tokens","rejection_rate","first_rejection_pos",
                    "rejection_concentration","rejection_severity","acceptance_decay_slope"]
        available = [c for c in rej_cols if c in kdf.columns]
        if available:
            gcols = ["workload","method","k","temperature"]
            gcols = [g for g in gcols if g in kdf.columns]
            out = safe_agg(kdf, gcols)
            write(out, out_dir / "aggregate_rejection_study.csv")
            # Rejection profile by workload × method (k=4 only for clean comparison)
            k4_rej = kdf[kdf["k"]==4].copy()
            if not k4_rej.empty:
                k4_out = safe_agg(k4_rej, ["workload","method","temperature"])
                write(k4_out, out_dir / "aggregate_rejection_k4.csv")

    # ── 14. Grand summary table (k=4, A_baseline) ─────────────────────────
    base = agg[agg["experiment"].str.startswith("A_baseline", na=False)].copy() \
           if "experiment" in agg.columns else agg.copy()
    if base.empty:
        base = agg.copy()
    k4 = base[base["k"] == 4].copy() if "k" in base.columns else base.copy()

    summary_cols = [
        "workload", "method", "k", "n",
        "tps_mean", "tps_std", "lat_mean", "lat_p95",
        "accept_rate_mean", "accepted_per_draft_mean",
        "accept_pos0_mean", "accept_pos3_mean",
        "rollback_freq_mean", "verif_util_mean",
        "entropy_mean", "rep_density_mean",
        # rejection metrics
        "rejection_rate_mean", "rejection_severity_mean",
        "first_rejection_pos_mean", "decay_slope_mean",
        # intra-sequence
        "entropy_w0_mean", "entropy_w2_mean", "entropy_trend_mean",
        # oracle
        "oracle_k_mean", "oracle_gain_mean",
        "speedup_vs_ar",
    ]
    summary_cols = [c for c in summary_cols if c in k4.columns]
    summary = k4[summary_cols].round(4)
    write(summary, out_dir / "grand_summary_table.csv")

    # ── Console summary ───────────────────────────────────────────────────
    print("\n── Grand summary (A_baseline, k=4) ──")
    disp_cols = ["workload", "method", "tps_mean", "accept_rate_mean",
                 "entropy_mean", "speedup_vs_ar"]
    disp_cols = [c for c in disp_cols if c in summary.columns]
    print(summary[disp_cols].to_string(index=False))


if __name__ == "__main__":
    main()