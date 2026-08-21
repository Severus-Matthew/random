#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


KNOWN_METHODS = {"ar", "ngram_sd", "draft_sd", "eagle3"}
EXPECTED_WORKLOADS = [
    "code_gen",
    "conversational_generation_gen",
    "conversational_generation_sft",
    "hardware_gen",
    "long_chain_reasoning",
    "long_context_completion",
    "long_horizon_swe",
    "mathematical_reasoning",
]


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def infer_meta_from_path(summary_path: Path, runs_root: Path) -> dict:
    rel = summary_path.relative_to(runs_root)
    parts = list(rel.parts)

    meta = {
        "source_summary_path": str(summary_path),
        "workload_path": parts[0] if len(parts) >= 1 else "unknown",
        "method_path": None,
        "k_path": np.nan,
        "temperature_path": np.nan,
    }

    lower_parts = [p.lower() for p in parts]

    for p in lower_parts:
        if p in KNOWN_METHODS:
            meta["method_path"] = p
            break

    # Prefer exact k folder like k1/k2/k4/k8/k16.
    for p in lower_parts:
        m = re.fullmatch(r"k(\d+|na)", p)
        if m:
            if m.group(1) == "na":
                meta["k_path"] = np.nan
            else:
                meta["k_path"] = int(m.group(1))
            break

    # Fallback from leaf name like k16_temp0.0_...
    if pd.isna(meta["k_path"]):
        m = re.search(r"(?:^|[_/-])k(\d+)(?:[_/.-]|$)", str(summary_path).lower())
        if m:
            meta["k_path"] = int(m.group(1))

    mt = re.search(r"temp([0-9]+(?:\.[0-9]+)?)", str(summary_path).lower())
    if mt:
        try:
            meta["temperature_path"] = float(mt.group(1))
        except Exception:
            pass

    return meta


def normalize_one_summary(p: Path, runs_root: Path) -> pd.DataFrame:
    meta = infer_meta_from_path(p, runs_root)

    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise RuntimeError(f"Could not read {p}: {e}") from e

    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Workload.
    if "workload" not in out.columns or out["workload"].isna().all():
        out["workload"] = meta["workload_path"]
    else:
        out["workload"] = out["workload"].fillna(meta["workload_path"])

    # Method.
    if "method" not in out.columns or out["method"].isna().all():
        out["method"] = meta["method_path"]
    else:
        out["method"] = out["method"].fillna(meta["method_path"])
    out["method"] = out["method"].astype(str).str.strip()

    # k.
    if "k" not in out.columns:
        out["k"] = meta["k_path"]
    else:
        out["k"] = safe_num(out["k"]).fillna(meta["k_path"])

    # For AR, ignore accidental k4 in path/run_name; AR has no speculative k.
    out.loc[out["method"].eq("ar"), "k"] = np.nan

    # Temperature.
    if "temperature" not in out.columns:
        out["temperature"] = meta["temperature_path"]
    else:
        out["temperature"] = safe_num(out["temperature"]).fillna(meta["temperature_path"])

    # Metric aliases.
    if "tokens_per_sec" not in out.columns and "tps" in out.columns:
        out["tokens_per_sec"] = out["tps"]
    if "latency_s" not in out.columns and "wall_s" in out.columns:
        out["latency_s"] = out["wall_s"]
    if "n_output_tokens" not in out.columns and "output_tokens" in out.columns:
        out["n_output_tokens"] = out["output_tokens"]

    for c in ["tokens_per_sec", "latency_s", "n_output_tokens", "draft_tokens",
              "accepted_tokens_total", "accepted_tokens", "rejected_tokens",
              "acceptance_rate", "rejection_rate", "accepted_tokens_per_verifier_pass"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = safe_num(out[c])

    # Normalize accepted column.
    out["accepted_tokens_norm"] = out["accepted_tokens_total"]
    out.loc[out["accepted_tokens_norm"].isna(), "accepted_tokens_norm"] = out["accepted_tokens"]

    # Recompute rejected tokens if missing.
    missing_rej = out["rejected_tokens"].isna() & out["draft_tokens"].notna() & out["accepted_tokens_norm"].notna()
    out.loc[missing_rej, "rejected_tokens"] = (
        out.loc[missing_rej, "draft_tokens"] - out.loc[missing_rej, "accepted_tokens_norm"]
    ).clip(lower=0)

    # For AR, speculative counters should be zero.
    ar_mask = out["method"].eq("ar")
    out.loc[ar_mask, ["draft_tokens", "accepted_tokens_norm", "rejected_tokens"]] = 0.0
    out.loc[ar_mask, "acceptance_rate"] = np.nan
    out.loc[ar_mask, "rejection_rate"] = np.nan

    # IDs.
    if "prompt_id" not in out.columns:
        if "id" in out.columns:
            out["prompt_id"] = out["id"]
        else:
            out["prompt_id"] = [f"{meta['workload_path']}_{i:05d}" for i in range(len(out))]

    out["source_summary_path"] = str(p)
    out["workload_path"] = meta["workload_path"]
    out["method_path"] = meta["method_path"]
    out["k_path"] = meta["k_path"]

    out["baseline_config"] = np.where(
        out["method"].eq("ar"),
        "ar",
        out["method"].astype(str) + "_k" + out["k"].astype("Int64").astype(str),
    )

    return out


def add_token_efficiency_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    for c in ["draft_tokens", "accepted_tokens", "rejected_tokens"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = safe_num(d[c])

    spec = d["method"].ne("ar")

    d["token_acceptance_pct"] = np.nan
    d["wasted_token_pct"] = np.nan

    denom = d["draft_tokens"].replace(0, np.nan)
    d.loc[spec, "token_acceptance_pct"] = 100.0 * d.loc[spec, "accepted_tokens"] / denom.loc[spec]
    d.loc[spec, "wasted_token_pct"] = 100.0 * d.loc[spec, "rejected_tokens"] / denom.loc[spec]

    # Table convention for AR.
    d.loc[~spec, "token_acceptance_pct"] = 100.0
    d.loc[~spec, "wasted_token_pct"] = 0.0

    return d


def summarize_request_rows(req: pd.DataFrame):
    d = req.copy()

    for c in ["tokens_per_sec", "latency_s", "n_output_tokens", "draft_tokens",
              "accepted_tokens_norm", "rejected_tokens", "accepted_tokens_per_verifier_pass"]:
        d[c] = safe_num(d[c])

    group_cols = ["method", "k", "baseline_config", "workload"]

    by_workload = (
        d.groupby(group_cols, dropna=False)
        .agg(
            n_requests=("tokens_per_sec", "size"),
            mean_tps=("tokens_per_sec", "mean"),
            median_tps=("tokens_per_sec", "median"),
            mean_latency_s=("latency_s", "mean"),
            median_latency_s=("latency_s", "median"),
            total_latency_s=("latency_s", "sum"),
            total_output_tokens=("n_output_tokens", "sum"),
            mean_output_tokens=("n_output_tokens", "mean"),
            draft_tokens=("draft_tokens", "sum"),
            accepted_tokens=("accepted_tokens_norm", "sum"),
            rejected_tokens=("rejected_tokens", "sum"),
            mean_accepted_tokens_per_verifier_pass=("accepted_tokens_per_verifier_pass", "mean"),
        )
        .reset_index()
    )

    by_workload["micro_tps_total_tokens_over_total_latency"] = (
        by_workload["total_output_tokens"] / by_workload["total_latency_s"].replace(0, np.nan)
    )
    by_workload = add_token_efficiency_cols(by_workload)

    # AR references by workload.
    ar_by_w = by_workload[by_workload["method"].eq("ar")][
        ["workload", "mean_tps", "mean_latency_s"]
    ].rename(columns={
        "mean_tps": "ar_mean_tps_workload",
        "mean_latency_s": "ar_mean_latency_s_workload",
    })

    by_workload = by_workload.merge(ar_by_w, on="workload", how="left")
    by_workload["throughput_speedup_vs_ar_workload"] = (
        by_workload["mean_tps"] / by_workload["ar_mean_tps_workload"].replace(0, np.nan)
    )
    by_workload["e2e_speedup_vs_ar_workload"] = (
        by_workload["ar_mean_latency_s_workload"] / by_workload["mean_latency_s"].replace(0, np.nan)
    )

    # Overall request-level micro: mean over all request rows.
    overall = (
        d.groupby(["method", "k", "baseline_config"], dropna=False)
        .agg(
            n_requests=("tokens_per_sec", "size"),
            n_workloads=("workload", "nunique"),
            mean_tps=("tokens_per_sec", "mean"),
            median_tps=("tokens_per_sec", "median"),
            mean_latency_s=("latency_s", "mean"),
            median_latency_s=("latency_s", "median"),
            total_latency_s=("latency_s", "sum"),
            total_output_tokens=("n_output_tokens", "sum"),
            mean_output_tokens=("n_output_tokens", "mean"),
            draft_tokens=("draft_tokens", "sum"),
            accepted_tokens=("accepted_tokens_norm", "sum"),
            rejected_tokens=("rejected_tokens", "sum"),
            mean_accepted_tokens_per_verifier_pass=("accepted_tokens_per_verifier_pass", "mean"),
        )
        .reset_index()
    )
    overall["micro_tps_total_tokens_over_total_latency"] = (
        overall["total_output_tokens"] / overall["total_latency_s"].replace(0, np.nan)
    )
    overall = add_token_efficiency_cols(overall)

    ar_overall = overall[overall["method"].eq("ar")]
    if ar_overall.empty:
        ar_mean_tps = np.nan
        ar_mean_latency = np.nan
    else:
        ar_mean_tps = float(ar_overall["mean_tps"].iloc[0])
        ar_mean_latency = float(ar_overall["mean_latency_s"].iloc[0])

    overall["ar_mean_tps_overall"] = ar_mean_tps
    overall["ar_mean_latency_s_overall"] = ar_mean_latency
    overall["throughput_speedup_vs_ar"] = overall["mean_tps"] / ar_mean_tps
    overall["e2e_speedup_vs_ar"] = ar_mean_latency / overall["mean_latency_s"].replace(0, np.nan)

    # Workload-macro table: average by-workload speedups and metrics equally across workloads.
    macro = (
        by_workload.groupby(["method", "k", "baseline_config"], dropna=False)
        .agg(
            n_workloads=("workload", "nunique"),
            macro_mean_tps=("mean_tps", "mean"),
            macro_mean_latency_s=("mean_latency_s", "mean"),
            macro_throughput_speedup_vs_ar=("throughput_speedup_vs_ar_workload", "mean"),
            macro_e2e_speedup_vs_ar=("e2e_speedup_vs_ar_workload", "mean"),
            macro_token_acceptance_pct=("token_acceptance_pct", "mean"),
            macro_wasted_token_pct=("wasted_token_pct", "mean"),
            macro_mean_accepted_tokens_per_verifier_pass=("mean_accepted_tokens_per_verifier_pass", "mean"),
        )
        .reset_index()
    )

    return overall, by_workload, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-root",
        default="results/apexp_block/config_large_500/evaluation/final_evaluations/runs",
    )
    ap.add_argument(
        "--out-dir",
        default="results/apexp_block/config_large_500/evaluation/final_evaluations/fixed_baseline_summary_from_runs",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        raise SystemExit(f"runs root not found: {runs_root}")

    summary_paths = sorted(runs_root.rglob("summary.csv"))

    frames = []
    errors = []

    for p in summary_paths:
        try:
            df = normalize_one_summary(p, runs_root)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            errors.append({"path": str(p), "error": repr(e)})

    if not frames:
        raise SystemExit(f"No readable summary.csv files found under {runs_root}")

    req = pd.concat(frames, ignore_index=True, sort=False)

    # Keep only known methods.
    req = req[req["method"].isin(KNOWN_METHODS)].copy()

    # Normalize numeric columns once more.
    for c in ["k", "temperature", "tokens_per_sec", "latency_s", "n_output_tokens",
              "draft_tokens", "accepted_tokens_norm", "rejected_tokens"]:
        if c in req.columns:
            req[c] = safe_num(req[c])

    overall, by_workload, macro = summarize_request_rows(req)

    # Sort.
    method_order = {"ar": 0, "ngram_sd": 1, "draft_sd": 2, "eagle3": 3}
    for df in [overall, by_workload, macro]:
        df["_method_order"] = df["method"].map(method_order).fillna(99)
        df["_k_sort"] = df["k"].fillna(-1)
        sort_cols = ["_method_order", "_k_sort"]
        if "workload" in df.columns:
            sort_cols = ["workload"] + sort_cols
        df.sort_values(sort_cols, inplace=True)
        df.drop(columns=["_method_order", "_k_sort"], inplace=True)

    # Final paper-facing table columns.
    overall_cols = [
        "method", "k", "baseline_config", "n_requests", "n_workloads",
        "mean_latency_s", "e2e_speedup_vs_ar",
        "mean_tps", "throughput_speedup_vs_ar",
        "token_acceptance_pct", "wasted_token_pct",
        "mean_accepted_tokens_per_verifier_pass",
        "draft_tokens", "accepted_tokens", "rejected_tokens",
        "mean_output_tokens",
    ]
    overall_cols = [c for c in overall_cols if c in overall.columns]

    by_workload_cols = [
        "workload", "method", "k", "baseline_config", "n_requests",
        "mean_latency_s", "ar_mean_latency_s_workload", "e2e_speedup_vs_ar_workload",
        "mean_tps", "ar_mean_tps_workload", "throughput_speedup_vs_ar_workload",
        "token_acceptance_pct", "wasted_token_pct",
        "mean_accepted_tokens_per_verifier_pass",
        "draft_tokens", "accepted_tokens", "rejected_tokens",
        "mean_output_tokens",
    ]
    by_workload_cols = [c for c in by_workload_cols if c in by_workload.columns]

    macro_cols = [
        "method", "k", "baseline_config", "n_workloads",
        "macro_mean_latency_s", "macro_e2e_speedup_vs_ar",
        "macro_mean_tps", "macro_throughput_speedup_vs_ar",
        "macro_token_acceptance_pct", "macro_wasted_token_pct",
        "macro_mean_accepted_tokens_per_verifier_pass",
    ]
    macro_cols = [c for c in macro_cols if c in macro.columns]

    # Write canonical files.
    req.to_csv(out_dir / "fixed_baseline_request_rows.csv", index=False)
    overall.to_csv(out_dir / "fixed_baseline_overall_micro.csv", index=False)
    by_workload.to_csv(out_dir / "fixed_baseline_by_workload_micro.csv", index=False)
    macro.to_csv(out_dir / "fixed_baseline_overall_macro_by_workload.csv", index=False)

    overall[overall_cols].to_csv(out_dir / "paper_fixed_table_overall.csv", index=False)
    by_workload[by_workload_cols].to_csv(out_dir / "paper_fixed_table_by_workload.csv", index=False)
    macro[macro_cols].to_csv(out_dir / "paper_fixed_table_macro_workload_matched.csv", index=False)

    # Inventory and warnings.
    inventory = {
        "runs_root": str(runs_root),
        "out_dir": str(out_dir),
        "n_summary_csv_found": len(summary_paths),
        "n_summary_csv_read": len(frames),
        "n_request_rows": int(len(req)),
        "workloads_found": sorted(req["workload"].dropna().astype(str).unique().tolist()),
        "methods_found": sorted(req["method"].dropna().astype(str).unique().tolist()),
        "configs_found": (
            req[["method", "k", "baseline_config"]]
            .drop_duplicates()
            .sort_values(["method", "k"], na_position="first")
            .to_dict(orient="records")
        ),
        "errors": errors[:50],
    }

    missing = sorted(set(EXPECTED_WORKLOADS) - set(inventory["workloads_found"]))
    extra = sorted(set(inventory["workloads_found"]) - set(EXPECTED_WORKLOADS))
    inventory["missing_expected_workloads"] = missing
    inventory["extra_workloads"] = extra

    (out_dir / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True))

    inv_rows = [{
        "n_summary_csv_found": len(summary_paths),
        "n_summary_csv_read": len(frames),
        "n_request_rows": int(len(req)),
        "n_workloads": req["workload"].nunique(),
        "workloads_found": ",".join(inventory["workloads_found"]),
        "missing_expected_workloads": ",".join(missing),
        "methods_found": ",".join(inventory["methods_found"]),
    }]
    pd.DataFrame(inv_rows).to_csv(out_dir / "00_inventory.csv", index=False)

    # Excel workbook.
    xlsx_path = out_dir / "fixed_baseline_tables_from_runs.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        pd.DataFrame(inv_rows).to_excel(w, sheet_name="inventory", index=False)
        overall[overall_cols].to_excel(w, sheet_name="overall_paper", index=False)
        by_workload[by_workload_cols].to_excel(w, sheet_name="by_workload_paper", index=False)
        macro[macro_cols].to_excel(w, sheet_name="macro_workload", index=False)
        overall.to_excel(w, sheet_name="overall_full", index=False)
        by_workload.to_excel(w, sheet_name="by_workload_full", index=False)

    print("WROTE:", out_dir)
    print("summary_csv_found:", len(summary_paths))
    print("request_rows:", len(req))
    print("workloads:", inventory["workloads_found"])
    print("methods:", inventory["methods_found"])
    if missing:
        print("[WARN] missing expected workloads:", missing)
    if errors:
        print("[WARN] errors while reading some summaries:", len(errors))
    print("MAIN TABLE:", out_dir / "paper_fixed_table_overall.csv")
    print("BY WORKLOAD:", out_dir / "paper_fixed_table_by_workload.csv")
    print("EXCEL:", xlsx_path)


if __name__ == "__main__":
    main()
