#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.apex_eval_utils import ensure_dir, save_bar_plot, write_json


METHODS = ["ar", "ngram_sd", "draft_sd", "eagle3", "ngram", "draft", "eagle"]


def infer_method_k_from_path(path: Path) -> dict:
    s = str(path)
    meta = {}
    for m in METHODS:
        if m in s:
            meta["method"] = {"ngram": "ngram_sd", "draft": "draft_sd", "eagle": "eagle3"}.get(m, m)
            break
    # Common patterns: k16, _k_16, k=16, num_speculative_tokens16.
    m = re.search(r"(?:^|[_/=-])k[_=]?([0-9]{1,2})(?:$|[_/.-])", s)
    if m:
        meta["k"] = int(m.group(1))
    else:
        m = re.search(r"num_speculative_tokens[_=]?([0-9]{1,2})", s)
        if m:
            meta["k"] = int(m.group(1))
    if "temperature" in s or "temp" in s:
        mt = re.search(r"temp(?:erature)?[_=]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", s)
        if mt:
            try:
                meta["temperature"] = float(mt.group(1).replace("p", "."))
            except Exception:
                pass
    return meta


def _iter_json_objects(obj: Any):
    """Yield dict objects from JSON files that may be dict, list[dict], or nested lists.

    Some result artifacts are plain request dicts, while others are list-valued
    manifests/debug outputs. The collector should skip irrelevant objects instead
    of crashing.
    """
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_json_objects(x)


def _looks_like_request_result(obj: dict) -> bool:
    # Hot run request result JSONs usually contain tokens_per_sec/wall_s.
    # Other manifests may contain only config/list data and should be ignored.
    return bool({"tokens_per_sec", "wall_s", "latency_s", "tps"} & set(obj.keys()))


def collect_result_jsons(root: Path) -> pd.DataFrame:
    rows = []
    skipped_non_dict = 0
    skipped_non_result = 0

    for p in root.rglob("*.json"):
        if "summary" in p.parts:
            continue
        try:
            raw = json.loads(p.read_text())
        except Exception:
            continue

        found_obj = False
        for obj in _iter_json_objects(raw):
            found_obj = True
            if not isinstance(obj, dict):
                skipped_non_dict += 1
                continue
            if not _looks_like_request_result(obj):
                skipped_non_result += 1
                continue
            meta = infer_method_k_from_path(p)
            row = {**obj, **meta, "source_path": str(p)}
            rows.append(row)

        if not found_obj:
            skipped_non_dict += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["skipped_non_dict"] = skipped_non_dict
        df.attrs["skipped_non_result"] = skipped_non_result
    return df


def summarize_requests(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    if "method" not in d.columns:
        d["method"] = "unknown"
    if "k" not in d.columns:
        d["k"] = np.nan
    if "workload" not in d.columns:
        d["workload"] = "unknown"
    if "temperature" not in d.columns:
        d["temperature"] = np.nan

    # Normalize possible metric aliases.
    if "tokens_per_sec" not in d.columns and "tps" in d.columns:
        d["tokens_per_sec"] = d["tps"]
    if "wall_s" not in d.columns and "latency_s" in d.columns:
        d["wall_s"] = d["latency_s"]

    d["tokens_per_sec"] = pd.to_numeric(d.get("tokens_per_sec"), errors="coerce")
    d["wall_s"] = pd.to_numeric(d.get("wall_s"), errors="coerce")
    d["output_tokens"] = pd.to_numeric(d.get("output_tokens"), errors="coerce")
    d["temperature"] = pd.to_numeric(d.get("temperature"), errors="coerce")

    group_cols = ["method", "k", "temperature", "workload"]
    out = d.groupby(group_cols, dropna=False).agg(
        n=("tokens_per_sec", "size"),
        mean_tps=("tokens_per_sec", "mean"),
        median_tps=("tokens_per_sec", "median"),
        mean_wall_s=("wall_s", "mean"),
        median_wall_s=("wall_s", "median"),
        mean_output_tokens=("output_tokens", "mean"),
    ).reset_index()
    return out.sort_values(["method", "k", "temperature", "workload"], na_position="last")


def collect_existing_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("request_summary_overall.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        meta = infer_method_k_from_path(p)
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
        df["summary_path"] = str(p)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Existing result roots to search recursively")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    plot_dir = ensure_dir(out_dir / "plots")

    all_json = []
    all_summaries = []
    for root_s in args.roots:
        root = Path(root_s)
        if not root.exists():
            print(f"[warn] missing root: {root}")
            continue
        rj = collect_result_jsons(root)
        if not rj.empty:
            rj["search_root"] = str(root)
            all_json.append(rj)
        sm = collect_existing_summaries(root)
        if not sm.empty:
            sm["search_root"] = str(root)
            all_summaries.append(sm)

    req = pd.concat(all_json, ignore_index=True) if all_json else pd.DataFrame()
    if not req.empty:
        req.to_csv(out_dir / "fixed_baseline_request_rows.csv", index=False)
        summary = summarize_requests(req)
        summary.to_csv(out_dir / "fixed_baseline_request_summary_by_method_k_temp_workload.csv", index=False)
        # Backward-compatible filename without temperature in the name.
        summary.to_csv(out_dir / "fixed_baseline_request_summary_by_method_k_workload.csv", index=False)
        overall = summary.groupby(["method", "k", "temperature"], dropna=False).agg(
            n=("n", "sum"),
            mean_tps=("mean_tps", "mean"),
            median_tps=("median_tps", "median"),
            mean_wall_s=("mean_wall_s", "mean"),
        ).reset_index().sort_values("mean_tps", ascending=False)
        overall.to_csv(out_dir / "fixed_baseline_request_summary_overall.csv", index=False)
        overall["label"] = (
            overall["method"].astype(str)
            + "_k" + overall["k"].astype(str)
            + "_t" + overall["temperature"].astype(str)
        )
        save_bar_plot(overall.head(40), "label", "mean_tps", plot_dir / "fixed_baseline_mean_tps_top40.png", "Fixed baseline mean TPS")
    else:
        summary = pd.DataFrame()
        overall = pd.DataFrame()

    sm = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    if not sm.empty:
        sm.to_csv(out_dir / "discovered_existing_request_summaries.csv", index=False)

    write_json(out_dir / "fixed_baseline_collection_manifest.json", {
        "roots": args.roots,
        "n_request_json_rows": int(len(req)),
        "n_existing_summary_rows": int(len(sm)),
        "note": "This collector is tolerant and recursive. It skips list-valued/config JSON artifacts unless they contain request-result dicts. For final paper tables, manually verify that discovered folders correspond to clean fixed-k baselines, not debug/smoke runs.",
    })

    print(f"request rows: {len(req)}")
    print(f"existing summary rows: {len(sm)}")
    print(f"WROTE: {out_dir}")


if __name__ == "__main__":
    main()
