#!/usr/bin/env python

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# ── All metrics to aggregate ──────────────────────────────────────────────
CORE_AGG = {
    "n":                      ("id",                                "count"),
    "tps_mean":               ("tokens_per_sec",                    "mean"),
    "tps_p50":                ("tokens_per_sec",                    "median"),
    "tps_std":                ("tokens_per_sec",                    "std"),
    "tps_p95":                ("tokens_per_sec",                    lambda x: x.quantile(0.95)),
    "lat_mean":               ("latency_s",                         "mean"),
    "lat_p50":                ("latency_s",                         "median"),
    "lat_p95":                ("latency_s",                         lambda x: x.quantile(0.95)),
    "lat_std":                ("latency_s",                         "std"),
    "n_tokens_mean":          ("n_output_tokens",                   "mean"),
    "accept_rate_mean":       ("acceptance_rate",                   "mean"),
    "accept_rate_std":        ("acceptance_rate",                   "std"),
    "accept_per_vpass_mean":  ("accepted_tokens_per_verifier_pass", "mean"),
    "draft_tokens_mean":      ("draft_tokens",                      "mean"),
    "accepted_tokens_mean":   ("accepted_tokens_total",             "mean"),
    "accept_pos0_mean":       ("accept_rate_pos_0",                 "mean"),
    "accept_pos1_mean":       ("accept_rate_pos_1",                 "mean"),
    "accept_pos2_mean":       ("accept_rate_pos_2",                 "mean"),
    "accept_pos3_mean":       ("accept_rate_pos_3",                 "mean"),
    "accepted_per_draft_mean":("accepted_per_draft",                "mean"),
    "rollback_freq_mean":     ("rollback_frequency",                "mean"),
    "rollback_freq_std":      ("rollback_frequency",                "std"),
    "verif_util_mean":        ("verifier_utilization",              "mean"),
    "entropy_mean":           ("mean_entropy",                      "mean"),
    "entropy_std":            ("mean_entropy",                      "std"),
    "rep_density_mean":       ("repetition_density",                "mean"),
    "acc_vol_mean":           ("acceptance_volatility",             "mean"),
    "ttft_mean":              ("ttft_s",                            "mean"),
    "itl_mean":               ("itl_s",                            "mean"),
    "prefill_mean":           ("prefill_time_s",                    "mean"),
    "decode_mean":            ("decode_time_s",                     "mean"),
    "tpot_mean":              ("tpot_s",                            "mean"),
    "prefix_cache_hit_rate":  ("prefix_cache_hit_rate",             "mean"),
    "prompt_cache_frac":      ("prompt_cache_frac",                 "mean"),
    "kv_cache_usage":         ("kv_cache_usage_perc",               "mean"),
    "rejected_tokens_mean":      ("rejected_tokens",          "mean"),
    "rejection_rate_mean":       ("rejection_rate",           "mean"),
    "rejection_rate_std":        ("rejection_rate",           "std"),
    "first_rejection_pos_mean":  ("first_rejection_pos",      "mean"),
    "rejection_conc_mean":       ("rejection_concentration",  "mean"),
    "rejection_severity_mean":   ("rejection_severity",       "mean"),
    "decay_slope_mean":          ("acceptance_decay_slope",   "mean"),
    "entropy_w0_mean":           ("entropy_w0_mean",          "mean"),
    "entropy_w1_mean":           ("entropy_w1_mean",          "mean"),
    "entropy_w2_mean":           ("entropy_w2_mean",          "mean"),
    "entropy_trend_mean":        ("entropy_trend",            "mean"),
    "entropy_range_mean":        ("entropy_range",            "mean"),
    "high_entropy_frac_mean":    ("high_entropy_frac",        "mean"),
    "oracle_k_mean":             ("oracle_k_estimated",       "mean"),
    "oracle_gain_mean":          ("oracle_k_vs_k4_gain",      "mean"),
    "oracle_expected_mean":      ("oracle_expected_tokens",   "mean"),
}

# Columns that, together with `id`, uniquely identify one prompt-run.
DEDUP_KEYS = [
    "experiment", "workload", "method", "k",
    "ngram_lookup_min", "ngram_lookup_max", "temperature",
    "max_prompt_tokens", "num_turns", "draft_model", "eagle3_model",
    "source_dataset", "id",
]

# Condition keys that define a comparable AR baseline. We always match on
# (workload, source_dataset); the others are added only when present so the
# baseline is as specific as the experiment allows.
BASELINE_KEYS = ["workload", "source_dataset", "temperature",
                 "max_prompt_tokens", "num_turns"]

# Lower number == lower priority. When the same run appears in two trees we
# keep the highest-priority copy. Adjust here if your canonical tree differs.
def _tree_priority(path_str: str) -> int:
    s = path_str.lower()
    if "phase2o" in s:      # stale / retry tree
        return 0
    if "/phase2/" in s or "\\phase2\\" in s:
        return 3
    if "/phase1/" in s or "\\phase1\\" in s:
        return 2
    return 1


def safe_agg(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Aggregate only columns that exist in df."""
    group_cols = [g for g in group_cols if g in df.columns]
    existing = {k: v for k, v in CORE_AGG.items() if v[0] in df.columns}
    if not existing:
        return df.groupby(group_cols, dropna=False).size().reset_index(name="n")
    return (df.groupby(group_cols, dropna=False)
              .agg(**{k: pd.NamedAgg(column=v[0], aggfunc=v[1])
                      for k, v in existing.items()})
              .reset_index())


def load_all(results_dir: Path, exclude: list = None) -> pd.DataFrame:
    """Recursively load all summary.csv files, de-duplicating across trees.

    `exclude` is a list of path substrings (case-insensitive). Any summary.csv
    whose path contains one of them is skipped entirely — use it to drop stale
    re-run trees such as `Phase2o`. De-duplication still runs as a safety net.
    """
    exclude = [e.lower() for e in (exclude or [])]
    all_paths = sorted(results_dir.glob("**/summary.csv"))
    paths = [p for p in all_paths
             if not any(e in str(p).lower() for e in exclude)]
    skipped = len(all_paths) - len(paths)
    if skipped:
        print(f"  Skipped {skipped} summary.csv from excluded trees: {exclude}")
    if not paths:
        raise SystemExit(f"No summary.csv found under {results_dir}")

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  [warn] could not read {p}: {e}")
            continue
        if df.empty:
            continue
        parts = p.parts
        if "experiment" not in df.columns:
            df["experiment"] = parts[-5] if len(parts) >= 5 else np.nan
        if "workload" not in df.columns:
            df["workload"] = parts[-4] if len(parts) >= 4 else "unknown"
        if "method" not in df.columns:
            df["method"] = parts[-3] if len(parts) >= 3 else "unknown"
        if "k" not in df.columns and len(parts) >= 2:
            k_str = str(parts[-2]).replace("k", "").split("_")[0]
            df["k"] = pd.to_numeric(k_str, errors="coerce")
        for col, default in [
            ("source_dataset", "unknown"),
            ("temperature", 0.0), ("max_prompt_tokens", 0), ("num_turns", 1),
            ("ngram_lookup_min", 1), ("ngram_lookup_max", 4),
            ("draft_model", ""), ("eagle3_model", ""),
            ("accept_rate_pos_0", np.nan), ("accept_rate_pos_1", np.nan),
            ("accept_rate_pos_2", np.nan), ("accept_rate_pos_3", np.nan),
            ("accepted_per_draft", np.nan), ("draft_tokens", np.nan),
            ("accepted_tokens_total", np.nan),
            ("ttft_s", np.nan), ("itl_s", np.nan), ("prefill_time_s", np.nan),
            ("decode_time_s", np.nan), ("tpot_s", np.nan),
            ("prefix_cache_hit_rate", np.nan), ("prompt_cache_frac", np.nan),
            ("kv_cache_usage_perc", np.nan),
        ]:
            if col not in df.columns:
                df[col] = default
        df["_src"] = str(p)
        df["_priority"] = _tree_priority(str(p))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    n_raw = len(combined)

    # Drop rows with no experiment label (stray files / corrupt configs).
    bad = combined["experiment"].isna() | (combined["experiment"].astype(str).str.strip() == "")
    if bad.any():
        print(f"  [warn] dropping {int(bad.sum())} row(s) with missing experiment")
        combined = combined[~bad].copy()

    # De-duplicate: keep highest-priority copy of each prompt-run.
    keys = [c for c in DEDUP_KEYS if c in combined.columns]
    if "id" in combined.columns:
        combined = (combined.sort_values("_priority", ascending=False)
                            .drop_duplicates(subset=keys, keep="first"))
    n_dedup = len(combined)
    if n_dedup != n_raw:
        print(f"  De-duplicated {n_raw} -> {n_dedup} rows "
              f"(removed {n_raw - n_dedup} stale/duplicate prompt-runs)")

    combined = combined.drop(columns=["_priority"], errors="ignore")
    print(f"  Loaded {len(combined)} rows from {len(paths)} files")
    return combined


def add_speedup_vs_ar(df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Attach speedup vs the AR baseline matched on data-source conditions.

    AR baselines are taken from `full_df` (the complete, de-duplicated run
    set) rather than from `df`, so studies that contain no AR rows of their
    own (e.g. B_k_sweep) still get a correct denominator from the experiment
    that did run AR on the same workload+dataset.
    """
    if "tps_mean" not in df.columns:
        return df

    keys = [k for k in BASELINE_KEYS if k in df.columns and k in full_df.columns]
    if "workload" not in keys:
        return df

    ar_rows = full_df[full_df["method"] == "ar"].copy()
    if ar_rows.empty:
        df["speedup_vs_ar"] = np.nan
        print("  [warn] no AR rows anywhere — speedup_vs_ar left NaN")
        return df

    ar = (ar_rows.groupby(keys, dropna=False)["tps_mean"]
                 .mean().rename("ar_tps").reset_index())

    merged = df.merge(ar, on=keys, how="left")
    # Fallback: if no exact condition match, match on (workload, source_dataset).
    if merged["ar_tps"].isna().any():
        coarse_keys = [k for k in ["workload", "source_dataset"] if k in keys]
        ar_coarse = (ar_rows.groupby(coarse_keys, dropna=False)["tps_mean"]
                            .mean().rename("ar_tps_coarse").reset_index())
        merged = merged.merge(ar_coarse, on=coarse_keys, how="left")
        merged["ar_tps"] = merged["ar_tps"].fillna(merged["ar_tps_coarse"])
        merged = merged.drop(columns=["ar_tps_coarse"], errors="ignore")

    missing = merged[(merged["method"] != "ar") & merged["ar_tps"].isna()]
    if not missing.empty:
        wls = sorted(missing["workload"].unique())
        print(f"  [warn] no matching AR baseline for: {', '.join(wls)} "
              f"-> speedup_vs_ar left NaN (run AR on that dataset to fix)")

    merged["speedup_vs_ar"] = merged["tps_mean"] / merged["ar_tps"].replace(0, np.nan)
    return merged.drop(columns=["ar_tps"], errors="ignore")


def write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  -> {path.name}  ({len(df)} rows)")


def _exp_mask(df, prefix):
    return df["experiment"].astype(str).str.startswith(prefix, na=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Path substrings to skip (stale re-run trees). "
                         "Default drops Phase2o/plotso.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {results_dir} ...")
    df = load_all(results_dir, exclude=args.exclude)
    df.drop(columns=["_src"], errors="ignore", inplace=True)

    # 1. Flat dump
    write(df, out_dir / "all_runs.csv")

    # 2. Baseline: experiment × workload × method × k  (speedup within experiment+dataset)
    agg = safe_agg(df, ["experiment", "workload", "method", "k", "source_dataset"])
    agg = add_speedup_vs_ar(agg, agg)
    write(agg, out_dir / "aggregate_by_workload_method.csv")

    # 3. k sweep (exp A + B), kept PER EXPERIMENT so synthetic & real don't pool
    ksweep = df[_exp_mask(df, "A_baseline") | _exp_mask(df, "B_k_sweep")].copy()
    if not ksweep.empty:
        out = safe_agg(ksweep, ["experiment", "workload", "method", "k", "source_dataset"])
        out = add_speedup_vs_ar(out, agg)
        write(out, out_dir / "aggregate_k_sweep.csv")

    # 4. Temperature sweep (exp E + A baseline at T=0)
    tdf = df[_exp_mask(df, "E_temperature") | _exp_mask(df, "A_baseline")].copy()
    if not tdf.empty and "temperature" in tdf.columns:
        out = safe_agg(tdf, ["workload", "method", "k", "temperature", "source_dataset"])
        out = add_speedup_vs_ar(out, agg)
        write(out, out_dir / "aggregate_temperature_sweep.csv")

    # 5. Context growth (exp F)
    cdf = df[_exp_mask(df, "F_context")].copy()
    if not cdf.empty and "max_prompt_tokens" in cdf.columns:
        out = safe_agg(cdf, ["workload", "method", "k", "max_prompt_tokens", "source_dataset"])
        out = add_speedup_vs_ar(out, agg)
        write(out, out_dir / "aggregate_context_growth.csv")

    # 6. Draft model sweep (exp D)
    ddf = df[_exp_mask(df, "D_draft")].copy()
    if not ddf.empty and "draft_model" in ddf.columns:
        out = safe_agg(ddf, ["workload", "method", "k", "draft_model", "source_dataset"])
        out = add_speedup_vs_ar(out, agg)
        write(out, out_dir / "aggregate_draft_model_sweep.csv")

    # 7. Ngram config sweep (exp C)
    ndf = df[_exp_mask(df, "C_ngram")].copy()
    if not ndf.empty:
        out = safe_agg(ndf, ["workload", "method", "k",
                             "ngram_lookup_min", "ngram_lookup_max", "source_dataset"])
        write(out, out_dir / "aggregate_ngram_config_sweep.csv")

    # 8. Multi-turn (exp G)
    mdf = df[_exp_mask(df, "G_multi")].copy()
    if not mdf.empty and "num_turns" in mdf.columns:
        out = safe_agg(mdf, ["workload", "method", "k", "num_turns", "source_dataset"])
        out = add_speedup_vs_ar(out, agg)
        write(out, out_dir / "aggregate_multi_turn.csv")

    # 9. Per-position acceptance profile
    pos_cols = [c for c in df.columns if c.startswith("accept_rate_pos_")]
    spec_df  = df[df["method"].isin(["eagle3", "draft_sd", "ngram_sd"])].copy()
    if pos_cols and not spec_df.empty:
        gcols = [c for c in ["experiment", "workload", "method", "k", "temperature"]
                 if c in spec_df.columns]
        pos_agg = spec_df.groupby(gcols, dropna=False)[pos_cols].mean().reset_index()
        pos_long = pos_agg.melt(id_vars=gcols, value_vars=pos_cols,
                                var_name="position", value_name="accept_rate_at_position")
        pos_long["position"] = pos_long["position"].str.extract(r"(\d+)").astype(float)
        write(pos_long, out_dir / "aggregate_positional_acceptance.csv")

    # 10. Oracle k study (exp H)
    hdf = df[_exp_mask(df, "H_oracle")].copy()
    if not hdf.empty:
        out = safe_agg(hdf, ["workload", "method", "k", "source_dataset"])
        out = add_speedup_vs_ar(out, agg)
        write(out, out_dir / "aggregate_oracle_k.csv")
        if "oracle_k_estimated" in hdf.columns and "entropy_bucket" in hdf.columns:
            oracle_detail = (hdf[hdf["method"].isin(["eagle3", "ngram_sd"])]
                             .groupby(["workload", "method", "k", "entropy_bucket"], dropna=False)
                             .agg(oracle_k_mean    =("oracle_k_estimated",  "mean"),
                                  oracle_gain_mean =("oracle_k_vs_k4_gain", "mean"),
                                  oracle_gain_pct  =("oracle_k_vs_k4_gain",
                                                     lambda x: (x > 0).mean() * 100),
                                  n                =("oracle_k_estimated",  "count"))
                             .reset_index())
            write(oracle_detail, out_dir / "aggregate_oracle_k_by_bucket.csv")

    # 11. Entropy bucket study (exp I)
    idf = df[_exp_mask(df, "I_entropy")].copy()
    if not idf.empty and "entropy_bucket" in idf.columns:
        out = safe_agg(idf, ["workload", "method", "k", "entropy_bucket", "source_dataset"])
        out = add_speedup_vs_ar(out, agg)
        write(out, out_dir / "aggregate_entropy_bucket.csv")
        if "tps_mean" in out.columns and "speedup_vs_ar" in out.columns:
            best_per_bucket = (out[out["method"] != "ar"]
                               .dropna(subset=["speedup_vs_ar"])
                               .sort_values("speedup_vs_ar", ascending=False)
                               .groupby(["workload", "entropy_bucket"], dropna=False)
                               .first().reset_index()
                               [["workload", "entropy_bucket", "method", "k",
                                 "tps_mean", "speedup_vs_ar"]])
            write(best_per_bucket, out_dir / "aggregate_entropy_bucket_best.csv")

    # 12. Intra-sequence transition (exp J)
    jdf = df[_exp_mask(df, "J_intra")].copy()
    if not jdf.empty:
        win_cols = [c for c in jdf.columns if c.startswith("entropy_w")]
        extra    = [c for c in ["entropy_trend", "entropy_range", "high_entropy_frac"]
                    if c in jdf.columns]
        cols = win_cols + extra
        if cols:
            out = (jdf.groupby(["workload", "method", "k"], dropna=False)
                      .agg(**{c: pd.NamedAgg(column=c, aggfunc="mean") for c in cols})
                      .reset_index())
            write(out, out_dir / "aggregate_intra_sequence.csv")

    # 13. Rejection study (exp K)
    kdf = df[_exp_mask(df, "K_rejection")].copy()
    if not kdf.empty:
        rej_cols = ["rejected_tokens", "rejection_rate", "first_rejection_pos",
                    "rejection_concentration", "rejection_severity", "acceptance_decay_slope"]
        if any(c in kdf.columns for c in rej_cols):
            out = safe_agg(kdf, ["workload", "method", "k", "temperature", "source_dataset"])
            write(out, out_dir / "aggregate_rejection_study.csv")
            k4_rej = kdf[kdf["k"] == 4].copy()
            if not k4_rej.empty:
                write(safe_agg(k4_rej, ["workload", "method", "temperature", "source_dataset"]),
                      out_dir / "aggregate_rejection_k4.csv")

    # 14. Grand summary table (k=4, A_baseline only — the canonical baseline run)
    base = agg[_exp_mask(agg, "A_baseline")].copy() if "experiment" in agg.columns else agg.copy()
    if base.empty:
        base = agg.copy()
    k4 = base[base["k"] == 4].copy() if "k" in base.columns else base.copy()
    summary_cols = [
        "experiment", "workload", "method", "k", "source_dataset", "n",
        "tps_mean", "tps_std", "lat_mean", "lat_p95",
        "accept_rate_mean", "accepted_per_draft_mean",
        "accept_pos0_mean", "accept_pos3_mean",
        "rollback_freq_mean", "verif_util_mean",
        "entropy_mean", "rep_density_mean",
        "rejection_rate_mean", "rejection_severity_mean",
        "first_rejection_pos_mean", "decay_slope_mean",
        "entropy_w0_mean", "entropy_w2_mean", "entropy_trend_mean",
        "oracle_k_mean", "oracle_gain_mean", "speedup_vs_ar",
    ]
    summary = k4[[c for c in summary_cols if c in k4.columns]].round(4)
    write(summary, out_dir / "grand_summary_table.csv")

    print("\n── Grand summary (A_baseline, k=4) ──")
    disp = [c for c in ["workload", "method", "source_dataset", "tps_mean",
                        "accept_rate_mean", "entropy_mean", "speedup_vs_ar"]
            if c in summary.columns]
    print(summary[disp].to_string(index=False))


if __name__ == "__main__":
    main()
