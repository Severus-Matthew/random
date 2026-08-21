#!/usr/bin/env python3
"""
Build Table 3: APEX ablation summary.

Expected usage:

python3 evaluation/make_table3_apex_ablation.py \
  --root results/apexp_block/config_large_500/evaluation/table3_apex_ablation_cap50_explicit_v1_cap50_7workloads \
  --fixed-summary-dir results/apexp_block/config_large_500/evaluation/final_evaluations/fixed_baseline_summary_from_runs

The script is intentionally robust to several summary layouts. It searches for:
  - request_summary_overall.csv
  - block_summary_overall.csv
  - summary.csv
inside --root.

It uses --fixed-summary-dir only to recover the AR reference latency/TPS.
Outputs are written to:
  <root>/table3_outputs/
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


# ----------------------------
# small helpers
# ----------------------------

def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        return float(x)
    except Exception:
        return default


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed reading {path}: {e}")
    return None


def _pick_pool_row(df: pd.DataFrame, preferred_pool: str = "slow_fast") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=object)

    if "pool" in df.columns:
        sub = df[df["pool"].astype(str).eq(preferred_pool)]
        if not sub.empty:
            return sub.iloc[0]

        # fallback: prefer learned/APEX-like rows before slow_only
        for p in ["fast", "apex", "learned", "slow_only"]:
            sub = df[df["pool"].astype(str).str.contains(p, case=False, na=False)]
            if not sub.empty:
                return sub.iloc[0]

    return df.iloc[0]


def infer_run_dir_from_summary(summary_file: Path) -> Path:
    # Typical: <run>/summary/request_summary_overall.csv
    if summary_file.parent.name == "summary":
        return summary_file.parent.parent
    return summary_file.parent


def infer_profile(run_name: str) -> str:
    s = run_name.lower()

    if "balanced" in s and "quality" not in s:
        return "APEX-Balanced"
    if "apex_balanced" in s:
        return "APEX-Balanced"
    if "efficient" in s:
        return "APEX-Efficient"
    if "apex_efficient" in s:
        return "APEX-Efficient"
    if "speed_first" in s or "apex_speed" in s or "speed" in s:
        return "APEX-Speed"

    # Runtime profile names
    if "quality_balanced" in s:
        return "APEX-Balanced"
    if "quality_strict" in s:
        return "APEX-Efficient"
    if "tps_strict" in s:
        return "APEX-Speed"

    return "APEX"


def infer_ablation(run_name: str) -> str:
    s = run_name.lower()

    ordered = [
        ("no_apex_depth", "w/o APEX-Depth"),
        ("without_apex_depth", "w/o APEX-Depth"),
        ("slow_only", "w/o APEX-Depth"),

        ("no_router", "w/o APEX-Router"),
        ("without_router", "w/o APEX-Router"),

        ("no_survival_loss", "w/o survival loss"),
        ("without_survival_loss", "w/o survival loss"),
        ("no_survival_accept_term", "w/o survival/acceptance term"),
        ("no_accept_term", "w/o survival/acceptance term"),

        ("no_cost_tps_term", "w/o cost/TPS term"),
        ("no_cost", "w/o cost/TPS term"),
        ("without_cost", "w/o cost/TPS term"),

        ("no_acceptance_rate_term", "w/o acceptance-rate term"),
        ("no_accept_rate", "w/o acceptance-rate term"),

        ("no_entropy_features", "w/o entropy features"),
        ("no_entropy", "w/o entropy features"),

        ("no_repetition_features", "w/o repetition features"),
        ("no_repetition", "w/o repetition features"),

        ("no_verifier_history_features", "w/o verifier-history features"),
        ("no_history", "w/o verifier-history features"),
        ("without_history", "w/o verifier-history features"),

        ("no_prompt_embedding", "w/o prompt embedding"),
        ("no_prompt_emb", "w/o prompt embedding"),

        ("no_uncertainty_features", "w/o uncertainty features"),
        ("no_uncertainty", "w/o uncertainty features"),
    ]

    for key, label in ordered:
        if key in s:
            return label

    if re.search(r"(^|[_\-/])base($|[_\-/])", s) or "full" in s:
        return "APEX full"

    return "APEX full"


def infer_candidate_set(run_name: str) -> str:
    s = run_name.lower()

    patterns = [
        ("k1_to_16", "{1,...,16}"),
        ("k1-16", "{1,...,16}"),
        ("k1_16", "{1,...,16}"),
        ("k3_to_16", "{3,...,16}"),
        ("k3-16", "{3,...,16}"),
        ("k3_16", "{3,...,16}"),
        ("k2_4_8_12_16", "{2,4,8,12,16}"),
        ("2_4_8_12_16", "{2,4,8,12,16}"),
    ]

    for key, val in patterns:
        if key in s:
            return val

    return ""


def ablation_order(label: str) -> int:
    order = {
        "APEX full": 0,
        "w/o APEX-Depth": 1,
        "w/o APEX-Router": 2,
        "w/o survival loss": 3,
        "w/o survival/acceptance term": 4,
        "w/o cost/TPS term": 5,
        "w/o acceptance-rate term": 6,
        "w/o entropy features": 7,
        "w/o repetition features": 8,
        "w/o verifier-history features": 9,
        "w/o prompt embedding": 10,
        "w/o uncertainty features": 11,
    }
    return order.get(label, 99)


# ----------------------------
# AR reference recovery
# ----------------------------

def recover_ar_reference(fixed_summary_dir: Optional[Path]) -> Dict[str, float]:
    out = {
        "ar_mean_tps": np.nan,
        "ar_mean_latency_s": np.nan,
        "source": "",
    }

    if fixed_summary_dir is None or not fixed_summary_dir.exists():
        print("[warn] fixed summary dir missing; speedup vs AR may be NaN")
        return out

    csvs = sorted(fixed_summary_dir.rglob("*.csv"))
    if not csvs:
        print("[warn] no CSV files found under fixed summary dir")
        return out

    # Best source: raw AR request rows.
    for p in csvs:
        if p.name == "ar_request_rows.csv" or "ar_request_rows" in p.name:
            df = _read_csv_safe(p)
            if df is None or df.empty:
                continue

            tps_col = _first_existing_col(df, ["tokens_per_sec", "tps", "mean_tps"])
            lat_col = _first_existing_col(df, ["latency_s", "latency", "mean_latency_s", "mean_latency"])

            if tps_col:
                out["ar_mean_tps"] = pd.to_numeric(df[tps_col], errors="coerce").mean()
            if lat_col:
                out["ar_mean_latency_s"] = pd.to_numeric(df[lat_col], errors="coerce").mean()

            out["source"] = str(p)
            return out

    # Second source: final fixed overall table with method == ar.
    for p in csvs:
        df = _read_csv_safe(p)
        if df is None or df.empty:
            continue

        method_col = _first_existing_col(df, ["method", "baseline_method", "baseline", "name"])
        if method_col is None:
            continue

        ar = df[df[method_col].astype(str).str.lower().eq("ar")]
        if ar.empty:
            ar = df[df[method_col].astype(str).str.lower().str.contains(r"(^|_)ar($|_)", regex=True, na=False)]

        if ar.empty:
            continue

        row = ar.iloc[0]

        tps_col = _first_existing_col(df, ["mean_tps", "tokens_per_sec", "tps", "ar_tps_reference"])
        lat_col = _first_existing_col(df, ["mean_latency_s", "mean_latency", "latency_s", "latency", "ar_latency_reference"])

        if tps_col:
            out["ar_mean_tps"] = _safe_float(row.get(tps_col))
        if lat_col:
            out["ar_mean_latency_s"] = _safe_float(row.get(lat_col))

        out["source"] = str(p)
        return out

    # Third source: any table carrying ar_tps_reference / ar_latency_reference.
    for p in csvs:
        df = _read_csv_safe(p)
        if df is None or df.empty:
            continue

        if "ar_tps_reference" in df.columns:
            vals = pd.to_numeric(df["ar_tps_reference"], errors="coerce").dropna()
            if len(vals):
                out["ar_mean_tps"] = vals.iloc[0]
                out["source"] = str(p)

        if "ar_latency_reference" in df.columns:
            vals = pd.to_numeric(df["ar_latency_reference"], errors="coerce").dropna()
            if len(vals):
                out["ar_mean_latency_s"] = vals.iloc[0]
                out["source"] = str(p)

        if math.isfinite(out["ar_mean_tps"]) or math.isfinite(out["ar_mean_latency_s"]):
            return out

    print("[warn] could not recover AR reference from fixed summary dir")
    return out


# ----------------------------
# APEX run parsing
# ----------------------------

def extract_one_run(run_dir: Path, request_file: Optional[Path], block_file: Optional[Path], summary_file: Optional[Path],
                    ar_ref: Dict[str, float]) -> Dict[str, Any]:
    run_name = run_dir.name

    req_row = pd.Series(dtype=object)
    blk_row = pd.Series(dtype=object)
    sum_row = pd.Series(dtype=object)

    req_df = _read_csv_safe(request_file) if request_file else None
    blk_df = _read_csv_safe(block_file) if block_file else None
    sum_df = _read_csv_safe(summary_file) if summary_file else None

    if req_df is not None:
        req_row = _pick_pool_row(req_df, "slow_fast")
    if blk_df is not None:
        blk_row = _pick_pool_row(blk_df, "slow_fast")
    if sum_df is not None and not sum_df.empty:
        sum_row = sum_df.iloc[0]

    def get_any(names, rows=(req_row, blk_row, sum_row), default=np.nan):
        for row in rows:
            for n in names:
                try:
                    if n in row.index:
                        v = row.get(n)
                        if pd.notna(v):
                            return v
                except Exception:
                    pass
        return default

    mean_tps = _safe_float(get_any(["mean_tps", "tokens_per_sec", "tps"]))
    mean_latency_s = _safe_float(get_any(["mean_latency_s", "mean_latency", "latency_s", "latency"]))

    n_results = _safe_float(get_any(["n_results", "n_requests", "num_requests", "n"]), default=np.nan)
    output_tokens = _safe_float(get_any(["output_tokens", "n_output_tokens", "total_output_tokens"]), default=np.nan)

    draft_tokens = _safe_float(get_any(["draft_tokens", "total_draft_tokens", "num_draft_tokens"]))
    accepted_tokens = _safe_float(get_any(["accepted_tokens", "accepted_tokens_total", "total_accepted_tokens"]))
    rejected_tokens = _safe_float(get_any(["rejected_tokens", "total_rejected_tokens"]))

    if not math.isfinite(rejected_tokens) and math.isfinite(draft_tokens) and math.isfinite(accepted_tokens):
        rejected_tokens = max(draft_tokens - accepted_tokens, 0.0)

    mean_k = _safe_float(get_any(["mean_k", "avg_k", "k_mean"]))
    accepted_per_block = _safe_float(get_any(["accepted_per_block", "mean_accepted", "avg_accepted_per_block"]))

    # Token acceptance / waste
    token_acceptance_pct = np.nan
    wasted_token_pct = np.nan

    if math.isfinite(draft_tokens) and draft_tokens > 0 and math.isfinite(accepted_tokens):
        token_acceptance_pct = 100.0 * accepted_tokens / draft_tokens

    if math.isfinite(draft_tokens) and draft_tokens > 0 and math.isfinite(rejected_tokens):
        wasted_token_pct = 100.0 * rejected_tokens / draft_tokens
    else:
        w_rate = _safe_float(get_any(["wasted_token_rate", "waste_rate"]))
        if math.isfinite(w_rate):
            wasted_token_pct = 100.0 * w_rate if w_rate <= 1.5 else w_rate

    if not math.isfinite(token_acceptance_pct) and math.isfinite(wasted_token_pct):
        token_acceptance_pct = 100.0 - wasted_token_pct

    # Speedup vs AR. Prefer E2E latency speedup if AR latency exists.
    ar_lat = ar_ref.get("ar_mean_latency_s", np.nan)
    ar_tps = ar_ref.get("ar_mean_tps", np.nan)

    e2e_speedup_vs_ar = np.nan
    throughput_speedup_vs_ar = np.nan

    if math.isfinite(ar_lat) and ar_lat > 0 and math.isfinite(mean_latency_s) and mean_latency_s > 0:
        e2e_speedup_vs_ar = ar_lat / mean_latency_s

    if math.isfinite(ar_tps) and ar_tps > 0 and math.isfinite(mean_tps):
        throughput_speedup_vs_ar = mean_tps / ar_tps

    # If a summary already has speedup columns, keep them as fallback.
    fallback_speed = _safe_float(get_any(["speedup_vs_ar", "latency_speedup_vs_ar", "e2e_speedup_vs_ar"]))
    if not math.isfinite(e2e_speedup_vs_ar) and math.isfinite(fallback_speed):
        e2e_speedup_vs_ar = fallback_speed

    if not math.isfinite(throughput_speedup_vs_ar):
        fallback_tps_speed = _safe_float(get_any(["throughput_speedup_vs_ar", "tps_speedup_vs_ar"]))
        if math.isfinite(fallback_tps_speed):
            throughput_speedup_vs_ar = fallback_tps_speed

    # WAS = Speedup / wasted-token fraction.
    # Use E2E speedup when available, otherwise throughput speedup.
    speed_for_was = e2e_speedup_vs_ar if math.isfinite(e2e_speedup_vs_ar) else throughput_speedup_vs_ar
    was = np.nan
    if math.isfinite(speed_for_was) and math.isfinite(wasted_token_pct) and wasted_token_pct > 0:
        was = speed_for_was / (wasted_token_pct / 100.0)

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "profile": infer_profile(run_name),
        "ablation": infer_ablation(run_name),
        "candidate_set": infer_candidate_set(run_name),

        "n_results": n_results,
        "output_tokens": output_tokens,

        "mean_latency_s": mean_latency_s,
        "mean_tps": mean_tps,

        "ar_latency_reference_s": ar_lat,
        "ar_tps_reference": ar_tps,

        "e2e_speedup_vs_ar": e2e_speedup_vs_ar,
        "throughput_speedup_vs_ar": throughput_speedup_vs_ar,

        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "rejected_tokens": rejected_tokens,
        "token_acceptance_pct": token_acceptance_pct,
        "wasted_token_pct": wasted_token_pct,

        "mean_k": mean_k,
        "accepted_per_block": accepted_per_block,

        "WAS": was,

        "request_summary_file": str(request_file) if request_file else "",
        "block_summary_file": str(block_file) if block_file else "",
        "summary_file": str(summary_file) if summary_file else "",
    }


def collect_apex_runs(root: Path, ar_ref: Dict[str, float]) -> pd.DataFrame:
    # Index files by run_dir.
    run_map: Dict[Path, Dict[str, Optional[Path]]] = {}

    for p in root.rglob("request_summary_overall.csv"):
        rd = infer_run_dir_from_summary(p)
        run_map.setdefault(rd, {})["request"] = p

    for p in root.rglob("block_summary_overall.csv"):
        rd = infer_run_dir_from_summary(p)
        run_map.setdefault(rd, {})["block"] = p

    # raw summary.csv fallback
    for p in root.rglob("summary.csv"):
        # Avoid fixed_baseline summary files if accidentally under root.
        if "fixed" in str(p).lower() and "apex" not in str(p).lower():
            continue
        rd = infer_run_dir_from_summary(p)
        run_map.setdefault(rd, {})["summary"] = p

    rows = []
    for rd, files in sorted(run_map.items(), key=lambda kv: str(kv[0])):
        rows.append(extract_one_run(
            rd,
            files.get("request"),
            files.get("block"),
            files.get("summary"),
            ar_ref,
        ))

    return pd.DataFrame(rows)


def add_full_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df["ablation_order"] = df["ablation"].map(ablation_order)

    df["delta_e2e_speedup_vs_full"] = np.nan
    df["delta_throughput_speedup_vs_full"] = np.nan
    df["delta_waste_pp_vs_full"] = np.nan
    df["delta_WAS_vs_full"] = np.nan

    for profile, sub in df.groupby("profile", dropna=False):
        full = sub[sub["ablation"].eq("APEX full")]
        if full.empty:
            # fallback: first row in this profile by order
            full = sub.sort_values(["ablation_order", "run_name"]).head(1)

        base = full.iloc[0]

        base_e2e = _safe_float(base.get("e2e_speedup_vs_ar"))
        base_tps = _safe_float(base.get("throughput_speedup_vs_ar"))
        base_waste = _safe_float(base.get("wasted_token_pct"))
        base_was = _safe_float(base.get("WAS"))

        idx = sub.index
        if math.isfinite(base_e2e):
            df.loc[idx, "delta_e2e_speedup_vs_full"] = df.loc[idx, "e2e_speedup_vs_ar"] - base_e2e
        if math.isfinite(base_tps):
            df.loc[idx, "delta_throughput_speedup_vs_full"] = df.loc[idx, "throughput_speedup_vs_ar"] - base_tps
        if math.isfinite(base_waste):
            df.loc[idx, "delta_waste_pp_vs_full"] = df.loc[idx, "wasted_token_pct"] - base_waste
        if math.isfinite(base_was):
            df.loc[idx, "delta_WAS_vs_full"] = df.loc[idx, "WAS"] - base_was

    return df


def make_latex_table(df: pd.DataFrame, out_tex: Path):
    if df.empty:
        out_tex.write_text("% No rows found.\n")
        return

    # One compact table. Prefer e2e speedup.
    cols = [
        "profile",
        "ablation",
        "e2e_speedup_vs_ar",
        "wasted_token_pct",
        "token_acceptance_pct",
        "WAS",
        "mean_k",
        "delta_e2e_speedup_vs_full",
        "delta_waste_pp_vs_full",
    ]

    use = df.copy()
    for c in cols:
        if c not in use.columns:
            use[c] = np.nan

    use = use[cols].copy()

    def fmt_x(v):
        return "--" if not math.isfinite(_safe_float(v)) else f"{float(v):.2f}$\\times$"

    def fmt_pct(v):
        return "--" if not math.isfinite(_safe_float(v)) else f"{float(v):.2f}"

    def fmt_num(v):
        return "--" if not math.isfinite(_safe_float(v)) else f"{float(v):.2f}"

    use["e2e_speedup_vs_ar"] = use["e2e_speedup_vs_ar"].map(fmt_x)
    use["wasted_token_pct"] = use["wasted_token_pct"].map(fmt_pct)
    use["token_acceptance_pct"] = use["token_acceptance_pct"].map(fmt_pct)
    use["WAS"] = use["WAS"].map(fmt_num)
    use["mean_k"] = use["mean_k"].map(fmt_num)
    use["delta_e2e_speedup_vs_full"] = use["delta_e2e_speedup_vs_full"].map(fmt_num)
    use["delta_waste_pp_vs_full"] = use["delta_waste_pp_vs_full"].map(fmt_num)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{APEX ablation study. Speedup is measured relative to autoregressive decoding. Waste is the percentage of drafted tokens rejected by the verifier. WAS denotes waste-amortized speedup, computed as speedup divided by the wasted-token fraction.}")
    lines.append(r"\label{tab:apex-ablation}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Profile} & \textbf{Variant} & \textbf{Speedup} & \textbf{Waste \%} & \textbf{Accept \%} & \textbf{WAS} & \textbf{Mean $k$} & \textbf{$\Delta$Speed} \\")
    lines.append(r"\midrule")

    for _, r in use.iterrows():
        lines.append(
            f"{r['profile']} & {r['ablation']} & {r['e2e_speedup_vs_ar']} & "
            f"{r['wasted_token_pct']} & {r['token_acceptance_pct']} & {r['WAS']} & "
            f"{r['mean_k']} & {r['delta_e2e_speedup_vs_full']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    out_tex.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--fixed-summary-dir", required=False, type=Path, default=None)
    ap.add_argument("--out-dir", required=False, type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    out_dir = args.out_dir or (root / "table3_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] root: {root}")
    print(f"[info] fixed summary dir: {args.fixed_summary_dir}")
    print(f"[info] out dir: {out_dir}")

    ar_ref = recover_ar_reference(args.fixed_summary_dir)
    print(f"[info] AR reference: {ar_ref}")

    df = collect_apex_runs((root / "apex_hot_runs"), ar_ref)
    if df.empty:
        print("[error] no APEX runs found. Expected request_summary_overall.csv, block_summary_overall.csv, or summary.csv under root.")
        print("[debug] first 100 files under root:")
        for p in sorted(root.rglob("*"))[:100]:
            print(" ", p)
        raise SystemExit(2)

    df = add_full_deltas(df)

    df = df.sort_values(
        ["profile", "ablation_order", "candidate_set", "run_name"],
        kind="stable",
    ).reset_index(drop=True)

    # Clean display table
    display_cols = [
        "profile",
        "ablation",
        "candidate_set",
        "n_results",
        "mean_latency_s",
        "mean_tps",
        "e2e_speedup_vs_ar",
        "throughput_speedup_vs_ar",
        "token_acceptance_pct",
        "wasted_token_pct",
        "WAS",
        "mean_k",
        "accepted_per_block",
        "delta_e2e_speedup_vs_full",
        "delta_waste_pp_vs_full",
        "run_name",
        "run_dir",
    ]

    for c in display_cols:
        if c not in df.columns:
            df[c] = np.nan

    clean = df[display_cols].copy()

    # Write outputs
    all_csv = out_dir / "table3_apex_ablation_all_runs.csv"
    clean_csv = out_dir / "table3_apex_ablation_clean.csv"
    tex_file = out_dir / "table3_apex_ablation.tex"
    ar_file = out_dir / "table3_ar_reference.txt"

    df.to_csv(all_csv, index=False)
    clean.to_csv(clean_csv, index=False)
    make_latex_table(clean, tex_file)
    ar_file.write_text(
        f"ar_mean_tps={ar_ref.get('ar_mean_tps')}\n"
        f"ar_mean_latency_s={ar_ref.get('ar_mean_latency_s')}\n"
        f"source={ar_ref.get('source')}\n"
    )

    print("\n=== Table 3 clean preview ===")
    preview_cols = [
        "profile",
        "ablation",
        "candidate_set",
        "e2e_speedup_vs_ar",
        "throughput_speedup_vs_ar",
        "wasted_token_pct",
        "token_acceptance_pct",
        "WAS",
        "mean_k",
        "delta_e2e_speedup_vs_full",
        "delta_waste_pp_vs_full",
    ]
    print(clean[preview_cols].to_string(index=False))

    print("\nWrote:")
    print(all_csv)
    print(clean_csv)
    print(tex_file)
    print(ar_file)


if __name__ == "__main__":
    main()
