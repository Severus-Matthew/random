#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {"ar", "ngram_sd", "draft_sd", "eagle3"}
POOLS = {"slow_fast", "slow_only"}


# -----------------------------
# Utilities
# -----------------------------
def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed reading {path}: {e}")
    return pd.DataFrame()


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_workload(x):
    if pd.isna(x):
        return "unknown"
    x = str(x)
    x = x.replace("_cap50", "")
    return x


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b.notna()) & (b != 0), a / b, np.nan)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_apex_run_name(run_name: str):
    parts = str(run_name).split("__")
    model_variant = parts[0] if len(parts) >= 1 else "unknown_model"
    profile = parts[1] if len(parts) >= 2 else "unknown_profile"
    candidate_set = parts[2] if len(parts) >= 3 else "unknown_kset"

    if "v4b" in model_variant:
        controller_family = "v4b_rankgroup"
    elif "v4" in model_variant:
        controller_family = "v4"
    else:
        controller_family = "unknown"

    return {
        "model_variant": model_variant,
        "profile": profile,
        "candidate_set": candidate_set,
        "controller_family": controller_family,
    }


def add_apex_metadata(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "run_name" not in df.columns:
        return df

    out = df.copy()

    if not manifest.empty and "run_name" in manifest.columns:
        keep = [
            c for c in [
                "run_name",
                "model_variant",
                "model_dir",
                "profile",
                "candidate_set",
                "candidate_ks",
                "min_k",
                "APEXP_ONLINE_SCORE_TPS_WEIGHT",
                "APEXP_ONLINE_SCORE_ACCEPT_RATE_WEIGHT",
                "APEXP_ONLINE_SCORE_WASTE_WEIGHT",
                "APEXP_ONLINE_SCORE_BLOCK_PENALTY_WEIGHT",
            ]
            if c in manifest.columns
        ]
        man = manifest[keep].drop_duplicates("run_name")
        for c in keep:
            if c != "run_name" and c in out.columns:
                out = out.drop(columns=[c])
        out = out.merge(man, on="run_name", how="left")

    parsed = pd.DataFrame([parse_apex_run_name(x) for x in out["run_name"]])
    for c in parsed.columns:
        if c not in out.columns or out[c].isna().all():
            out[c] = parsed[c]
        else:
            out[c] = out[c].fillna(parsed[c])

    return out


def infer_fixed_config(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "workload" in out.columns:
        out["workload"] = out["workload"].map(normalize_workload)

    if "method" not in out.columns:
        out["method"] = "unknown"

    if "k" not in out.columns:
        # Try extracting from run/config strings.
        src = None
        for c in ["config", "run_name", "path", "run_dir"]:
            if c in out.columns:
                src = c
                break
        if src:
            out["k"] = out[src].astype(str).str.extract(r"k(\d+)")[0]
        else:
            out["k"] = np.nan

    out["k"] = pd.to_numeric(out["k"], errors="coerce")

    def make_config(row):
        m = str(row.get("method", "unknown"))
        k = row.get("k", np.nan)
        if pd.notna(k):
            return f"{m}_k{int(k)}"
        return m

    out["baseline_config"] = out.apply(make_config, axis=1)
    return out


def iter_jsonl(path: Path):
    try:
        with path.open("r", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue

                # Some traces/log views contain literal \n between JSON objects.
                if "}\\n{" in raw:
                    parts = raw.split("\\n")
                else:
                    parts = [raw]

                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        yield json.loads(part)
                    except Exception:
                        continue
    except Exception:
        return


def flatten_state(row):
    out = {}
    state = row.get("state")
    if isinstance(state, dict):
        for k, v in state.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[f"state_{k}"] = v
    for k, v in row.items():
        if k == "state":
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


# -----------------------------
# Fixed baseline loading
# -----------------------------
def load_fixed_request_rows(eval_root: Path) -> pd.DataFrame:
    summary_dir = eval_root / "fixed_baseline_summary"
    p = summary_dir / "fixed_baseline_request_rows.csv"
    df = read_csv_or_empty(p)

    if not df.empty:
        df["source_table"] = str(p)
        return infer_fixed_config(df)

    # Fallback: scan fixed_runs summary.csv
    rows = []
    fixed_root = eval_root / "fixed_runs"
    for sp in fixed_root.rglob("summary.csv"):
        try:
            tmp = pd.read_csv(sp)
        except Exception:
            continue

        parts = sp.parts
        workload = "unknown"
        method = "unknown"

        for x in parts:
            if x in METHODS:
                method = x
                break

        # fixed_runs/<workload>/<method>/<config>/summary.csv
        if "fixed_runs" in parts:
            idx = parts.index("fixed_runs")
            if idx + 1 < len(parts):
                workload = parts[idx + 1]

        config = sp.parent.name
        tmp["workload"] = tmp.get("workload", workload)
        tmp["method"] = tmp.get("method", method)
        tmp["run_dir"] = str(sp.parent)
        tmp["config"] = config
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    return infer_fixed_config(df)


def aggregate_request_rows(df: pd.DataFrame, group_cols, ar_tps=None, workload_ar=None):
    if df.empty:
        return pd.DataFrame()

    x = df.copy()

    tps_col = pick_col(x, ["tokens_per_sec", "mean_tps", "tps"])
    lat_col = pick_col(x, ["latency_s", "mean_wall_s", "wall_s", "e2e_latency_engine_s"])
    out_tok_col = pick_col(x, ["n_output_tokens", "output_tokens", "mean_output_tokens"])
    draft_col = pick_col(x, ["draft_tokens", "num_draft_tokens"])
    acc_col = pick_col(x, ["accepted_tokens_total", "accepted_tokens", "accepted"])
    rej_col = pick_col(x, ["rejected_tokens", "num_rejected"])

    for c in [tps_col, lat_col, out_tok_col, draft_col, acc_col, rej_col]:
        if c and c in x.columns:
            x[c] = to_num(x[c])

    # Convert common rate columns.
    for c in [
        "acceptance_rate",
        "rejection_rate",
        "accepted_per_draft",
        "accepted_tokens_per_verifier_pass",
        "mean_entropy",
        "repetition_density",
        "first_rejection_pos",
        "acceptance_decay_slope",
        "rejection_concentration",
        "rejection_severity",
        "entropy_range",
        "high_entropy_frac",
        "prefix_cache_hit_rate",
        "kv_cache_usage_perc",
        "ttft_s",
        "itl_s",
        "tpot_s",
    ]:
        if c in x.columns:
            x[c] = to_num(x[c])

    aggs = {"n_requests": (tps_col, "count")} if tps_col else {"n_requests": (group_cols[0], "count")}

    if tps_col:
        aggs.update({
            "mean_tps": (tps_col, "mean"),
            "median_tps": (tps_col, "median"),
            "p05_tps": (tps_col, lambda s: s.quantile(0.05)),
            "p95_tps": (tps_col, lambda s: s.quantile(0.95)),
        })

    if lat_col:
        aggs.update({
            "mean_latency_s": (lat_col, "mean"),
            "median_latency_s": (lat_col, "median"),
            "p95_latency_s": (lat_col, lambda s: s.quantile(0.95)),
        })

    if out_tok_col:
        aggs["output_tokens"] = (out_tok_col, "sum")

    if draft_col:
        aggs["draft_tokens"] = (draft_col, "sum")

    if acc_col:
        aggs["accepted_tokens"] = (acc_col, "sum")

    if rej_col:
        aggs["rejected_tokens"] = (rej_col, "sum")

    for c in [
        "acceptance_rate",
        "rejection_rate",
        "accepted_per_draft",
        "accepted_tokens_per_verifier_pass",
        "mean_entropy",
        "repetition_density",
        "first_rejection_pos",
        "acceptance_decay_slope",
        "rejection_concentration",
        "rejection_severity",
        "entropy_range",
        "high_entropy_frac",
        "prefix_cache_hit_rate",
        "kv_cache_usage_perc",
        "ttft_s",
        "itl_s",
        "tpot_s",
    ]:
        if c in x.columns:
            aggs[f"mean_{c}"] = (c, "mean")

    out = x.groupby(group_cols, dropna=False).agg(**aggs).reset_index()

    if "draft_tokens" in out.columns and "accepted_tokens" in out.columns:
        if "rejected_tokens" not in out.columns:
            out["rejected_tokens"] = out["draft_tokens"] - out["accepted_tokens"]
        out["token_acceptance_rate"] = safe_div(out["accepted_tokens"], out["draft_tokens"])
        out["token_acceptance_pct"] = 100 * out["token_acceptance_rate"]
        out["wasted_token_rate"] = safe_div(out["rejected_tokens"], out["draft_tokens"])
        out["wasted_token_pct"] = 100 * out["wasted_token_rate"]

    if "accepted_tokens" in out.columns and "n_requests" in out.columns:
        out["accepted_tokens_per_request"] = safe_div(out["accepted_tokens"], out["n_requests"])

    if "rejected_tokens" in out.columns and "n_requests" in out.columns:
        out["wasted_tokens_per_request"] = safe_div(out["rejected_tokens"], out["n_requests"])

    if "mean_tps" in out.columns:
        if "workload" in out.columns and workload_ar:
            out["ar_tps_reference"] = out["workload"].map(workload_ar)
            out["speedup_vs_ar"] = safe_div(out["mean_tps"], out["ar_tps_reference"])
        elif ar_tps:
            out["ar_tps_reference"] = ar_tps
            out["speedup_vs_ar"] = out["mean_tps"] / ar_tps

    if "speedup_vs_ar" in out.columns and "wasted_token_rate" in out.columns:
        out["STE_speed_adjusted_token_efficiency"] = out["speedup_vs_ar"] * (1 - out["wasted_token_rate"])

    if "speedup_vs_ar" in out.columns and "wasted_token_rate" in out.columns:
        apb_col = pick_col(out, ["accepted_tokens_per_request", "mean_accepted_tokens_per_verifier_pass"])
        if apb_col:
            out["ATE_accepted_throughput_efficiency"] = (
                out["speedup_vs_ar"] * out[apb_col] * (1 - out["wasted_token_rate"])
            )

    return out


def get_ar_references(fixed_req: pd.DataFrame):
    if fixed_req.empty:
        return np.nan, {}

    x = fixed_req.copy()
    if "method" not in x.columns:
        return np.nan, {}

    method = x["method"].astype(str).str.lower()
    ar = x[method.eq("ar") | method.eq("baseline_ar")].copy()

    if ar.empty:
        # Some pipelines encode AR as k=1 with method none.
        if "baseline_config" in x.columns:
            ar = x[x["baseline_config"].astype(str).str.lower().isin(["ar", "baseline_ar"])].copy()

    if ar.empty:
        return np.nan, {}

    tps_col = pick_col(ar, ["tokens_per_sec", "mean_tps", "tps"])
    if not tps_col:
        return np.nan, {}

    ar[tps_col] = to_num(ar[tps_col])
    overall = float(ar[tps_col].mean())

    workload_ar = {}
    if "workload" in ar.columns:
        workload_ar = ar.groupby("workload")[tps_col].mean().to_dict()

    return overall, workload_ar


# -----------------------------
# APEX summary loading
# -----------------------------
def load_manifest(eval_root: Path) -> pd.DataFrame:
    candidates = [
        eval_root / "profile_kgrid_manifest.json",
        eval_root / "three_profile_manifest.json",
        eval_root.parent / "profile_kgrid_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.DataFrame(json.loads(p.read_text()))
            except Exception as e:
                print(f"[warn] could not read manifest {p}: {e}")
    return pd.DataFrame()


def load_apex_summary_tables(eval_root: Path, manifest: pd.DataFrame):
    sdir = eval_root / "apex_hot_summary"

    req_overall = read_csv_or_empty(sdir / "request_overall_all_runs.csv")
    if req_overall.empty:
        req_overall = read_csv_or_empty(sdir / "request_summary_overall_all_runs.csv")

    block_overall = read_csv_or_empty(sdir / "block_overall_all_runs.csv")
    if block_overall.empty:
        block_overall = read_csv_or_empty(sdir / "block_summary_overall_all_runs.csv")

    req_workload = read_csv_or_empty(sdir / "request_by_workload_all_runs.csv")
    if req_workload.empty:
        req_workload = read_csv_or_empty(sdir / "request_summary_by_workload_all_runs.csv")

    block_workload = read_csv_or_empty(sdir / "block_by_workload_all_runs.csv")
    if block_workload.empty:
        block_workload = read_csv_or_empty(sdir / "block_summary_by_workload_all_runs.csv")

    kdist = read_csv_or_empty(sdir / "active_k_distribution_all_runs.csv")

    for name, df in [
        ("req_overall", req_overall),
        ("block_overall", block_overall),
        ("req_workload", req_workload),
        ("block_workload", block_workload),
        ("kdist", kdist),
    ]:
        if not df.empty and "workload" in df.columns:
            df["workload"] = df["workload"].map(normalize_workload)

    req_overall = add_apex_metadata(req_overall, manifest)
    block_overall = add_apex_metadata(block_overall, manifest)
    req_workload = add_apex_metadata(req_workload, manifest)
    block_workload = add_apex_metadata(block_workload, manifest)
    kdist = add_apex_metadata(kdist, manifest)

    return req_overall, block_overall, req_workload, block_workload, kdist


def merge_apex_request_block(req: pd.DataFrame, block: pd.DataFrame, keys):
    if req.empty and block.empty:
        return pd.DataFrame()

    if req.empty:
        out = block.copy()
    elif block.empty:
        out = req.copy()
    else:
        keys = [k for k in keys if k in req.columns and k in block.columns]
        r = req.copy()
        b = block.copy()
        out = r.merge(b, on=keys, how="outer", suffixes=("_request", "_block"))

    # Normalize key metric names.
    for c in ["draft_tokens", "accepted_tokens", "rejected_tokens", "n_blocks"]:
        bc = f"{c}_block"
        if bc in out.columns and c not in out.columns:
            out[c] = out[bc]

    # TPS may come from request or block table.
    if "mean_tps_request" in out.columns:
        out["mean_tps"] = out["mean_tps_request"]
    elif "mean_tps" not in out.columns and "mean_tps_block" in out.columns:
        out["mean_tps"] = out["mean_tps_block"]

    if "n_request" in out.columns and "n_requests" not in out.columns:
        out["n_requests"] = out["n_request"]
    if "n" in out.columns and "n_requests" not in out.columns:
        out["n_requests"] = out["n"]

    for c in [
        "draft_tokens",
        "accepted_tokens",
        "rejected_tokens",
        "mean_tps",
        "mean_k",
        "mean_accepted_len",
        "accepted_per_block",
        "full_accept_rate",
        "n_blocks",
        "n_requests",
    ]:
        if c in out.columns:
            out[c] = to_num(out[c])

    if "draft_tokens" in out.columns and "accepted_tokens" in out.columns:
        if "rejected_tokens" not in out.columns:
            out["rejected_tokens"] = out["draft_tokens"] - out["accepted_tokens"]
        out["token_acceptance_rate"] = safe_div(out["accepted_tokens"], out["draft_tokens"])
        out["token_acceptance_pct"] = 100 * out["token_acceptance_rate"]
        out["wasted_token_rate"] = safe_div(out["rejected_tokens"], out["draft_tokens"])
        out["wasted_token_pct"] = 100 * out["wasted_token_rate"]

    if "accepted_tokens" in out.columns and "n_blocks" in out.columns:
        out["accepted_per_block_from_totals"] = safe_div(out["accepted_tokens"], out["n_blocks"])

    return out


def add_speedups(df: pd.DataFrame, ar_tps, workload_ar):
    out = df.copy()
    if out.empty or "mean_tps" not in out.columns:
        return out

    if "workload" in out.columns and workload_ar:
        out["ar_tps_reference"] = out["workload"].map(workload_ar)
        out["speedup_vs_ar"] = safe_div(out["mean_tps"], out["ar_tps_reference"])
    elif pd.notna(ar_tps) and ar_tps:
        out["ar_tps_reference"] = ar_tps
        out["speedup_vs_ar"] = out["mean_tps"] / ar_tps

    if "speedup_vs_ar" in out.columns and "wasted_token_rate" in out.columns:
        out["STE_speed_adjusted_token_efficiency"] = out["speedup_vs_ar"] * (1 - out["wasted_token_rate"])

    apb_col = pick_col(out, ["accepted_per_block", "accepted_per_block_from_totals", "mean_accepted_len"])
    if "speedup_vs_ar" in out.columns and "wasted_token_rate" in out.columns and apb_col:
        out["ATE_accepted_throughput_efficiency"] = (
            out["speedup_vs_ar"] * out[apb_col] * (1 - out["wasted_token_rate"])
        )

    return out


# -----------------------------
# Raw APEX scanning for k/workload and feature distributions
# -----------------------------
def infer_pool_method_from_path(path: Path):
    parts = list(path.parts)
    pool = "unknown_pool"
    method = "unknown_method"
    for i, x in enumerate(parts):
        if x in POOLS:
            pool = x
            if i + 1 < len(parts):
                method = parts[i + 1]
            break
    return pool, method


def bucket_entropy(x):
    try:
        x = float(x)
    except Exception:
        return "unknown"
    if not math.isfinite(x):
        return "unknown"
    if x < 1:
        return "[0,1)"
    if x < 2:
        return "[1,2)"
    if x < 3:
        return "[2,3)"
    if x < 4:
        return "[3,4)"
    if x < 5:
        return "[4,5)"
    if x < 6:
        return "[5,6)"
    return "[6,+)"


def bucket_repetition(x):
    try:
        x = float(x)
    except Exception:
        return "unknown"
    if not math.isfinite(x):
        return "unknown"
    if x < 0.02:
        return "[0,0.02)"
    if x < 0.05:
        return "[0.02,0.05)"
    if x < 0.10:
        return "[0.05,0.10)"
    if x < 0.20:
        return "[0.10,0.20)"
    if x < 0.40:
        return "[0.20,0.40)"
    return "[0.40,+)"


def find_numeric_feature(flat, names):
    for name in names:
        if name in flat:
            try:
                v = float(flat[name])
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    # substring fallback
    for k, v in flat.items():
        kl = k.lower()
        if any(n.lower() in kl for n in names):
            try:
                v = float(v)
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return np.nan


def scan_apex_raw(eval_root: Path, manifest: pd.DataFrame):
    runs_root = eval_root / "apex_hot_runs"
    if not runs_root.exists():
        print(f"[warn] APEX raw run root not found: {runs_root}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    k_counter = Counter()
    k_work_counter = Counter()
    entropy_counter = defaultdict(lambda: defaultdict(float))
    repetition_counter = defaultdict(lambda: defaultdict(float))
    feature_acc = defaultdict(lambda: defaultdict(float))

    n_files = 0
    n_events = 0

    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        run_name = run_dir.name
        meta = parse_apex_run_name(run_name)

        for bp in run_dir.rglob("block_events.jsonl"):
            n_files += 1
            pool, method = infer_pool_method_from_path(bp)

            for row in iter_jsonl(bp):
                flat = flatten_state(row)

                if "accepted_len" not in flat and "k_actual" not in flat and "active_k" not in flat:
                    continue

                workload = normalize_workload(
                    flat.get("workload")
                    or flat.get("prompt_id", "unknown").split("_cap50")[0]
                    or "unknown"
                )

                try:
                    k = int(float(flat.get("active_k", flat.get("k_actual", flat.get("k_requested", np.nan)))))
                except Exception:
                    k = None

                try:
                    accepted_len = float(flat.get("accepted_len", np.nan))
                except Exception:
                    accepted_len = np.nan

                try:
                    k_actual = float(flat.get("k_actual", k if k is not None else np.nan))
                except Exception:
                    k_actual = np.nan

                try:
                    rejected = float(flat.get("num_rejected", np.nan))
                except Exception:
                    rejected = np.nan

                if math.isnan(rejected) and math.isfinite(k_actual) and math.isfinite(accepted_len):
                    rejected = max(k_actual - accepted_len, 0)

                if k is not None:
                    key = (
                        run_name,
                        pool,
                        method,
                        meta["model_variant"],
                        meta["profile"],
                        meta["candidate_set"],
                        k,
                    )
                    k_counter[key] += 1

                    keyw = (
                        run_name,
                        pool,
                        method,
                        workload,
                        meta["model_variant"],
                        meta["profile"],
                        meta["candidate_set"],
                        k,
                    )
                    k_work_counter[keyw] += 1

                entropy = find_numeric_feature(
                    flat,
                    [
                        "mean_entropy",
                        "entropy",
                        "prefix_entropy",
                        "state_mean_entropy",
                        "state_prefix_entropy",
                        "state_entropy",
                    ],
                )
                rep = find_numeric_feature(
                    flat,
                    [
                        "repetition_density",
                        "repetition",
                        "repeat",
                        "prefix_repetition",
                        "state_repetition_density",
                        "state_prefix_repetition",
                    ],
                )

                common_key = (
                    run_name,
                    pool,
                    method,
                    workload,
                    meta["model_variant"],
                    meta["profile"],
                    meta["candidate_set"],
                )

                if math.isfinite(entropy):
                    b = bucket_entropy(entropy)
                    ek = common_key + (b,)
                    entropy_counter[ek]["n_blocks"] += 1
                    if math.isfinite(k_actual):
                        entropy_counter[ek]["sum_k"] += k_actual
                    if math.isfinite(accepted_len):
                        entropy_counter[ek]["accepted_tokens"] += accepted_len
                    if math.isfinite(rejected):
                        entropy_counter[ek]["rejected_tokens"] += rejected
                    if math.isfinite(k_actual):
                        entropy_counter[ek]["draft_tokens"] += k_actual

                if math.isfinite(rep):
                    b = bucket_repetition(rep)
                    rk = common_key + (b,)
                    repetition_counter[rk]["n_blocks"] += 1
                    if math.isfinite(k_actual):
                        repetition_counter[rk]["sum_k"] += k_actual
                    if math.isfinite(accepted_len):
                        repetition_counter[rk]["accepted_tokens"] += accepted_len
                    if math.isfinite(rejected):
                        repetition_counter[rk]["rejected_tokens"] += rejected
                    if math.isfinite(k_actual):
                        repetition_counter[rk]["draft_tokens"] += k_actual

                # Generic feature means for paper mining.
                for fname, val in flat.items():
                    fl = fname.lower()
                    if not any(s in fl for s in ["entropy", "repeat", "repetition", "accept", "reject", "zero", "full"]):
                        continue
                    try:
                        val = float(val)
                    except Exception:
                        continue
                    if not math.isfinite(val):
                        continue
                    fk = common_key + (fname,)
                    feature_acc[fk]["sum_value"] += val
                    feature_acc[fk]["n"] += 1

                n_events += 1

    print(f"[raw scan] block_events files={n_files}, usable_events={n_events}")

    k_rows = []
    for key, n in k_counter.items():
        run_name, pool, method, model_variant, profile, candidate_set, k = key
        k_rows.append({
            "run_name": run_name,
            "pool": pool,
            "method": method,
            "model_variant": model_variant,
            "profile": profile,
            "candidate_set": candidate_set,
            "active_k": k,
            "n_blocks": n,
        })
    k_df = pd.DataFrame(k_rows)
    if not k_df.empty:
        k_df["pct_within_run_pool_method"] = (
            k_df["n_blocks"] / k_df.groupby(["run_name", "pool", "method"])["n_blocks"].transform("sum")
        )

    kw_rows = []
    for key, n in k_work_counter.items():
        run_name, pool, method, workload, model_variant, profile, candidate_set, k = key
        kw_rows.append({
            "run_name": run_name,
            "pool": pool,
            "method": method,
            "workload": workload,
            "model_variant": model_variant,
            "profile": profile,
            "candidate_set": candidate_set,
            "active_k": k,
            "n_blocks": n,
        })
    kw_df = pd.DataFrame(kw_rows)
    if not kw_df.empty:
        kw_df["pct_within_run_pool_method_workload"] = (
            kw_df["n_blocks"] /
            kw_df.groupby(["run_name", "pool", "method", "workload"])["n_blocks"].transform("sum")
        )

    def counter_to_df(counter, bucket_name):
        rows = []
        for key, vals in counter.items():
            run_name, pool, method, workload, model_variant, profile, candidate_set, bucket = key
            row = {
                "run_name": run_name,
                "pool": pool,
                "method": method,
                "workload": workload,
                "model_variant": model_variant,
                "profile": profile,
                "candidate_set": candidate_set,
                bucket_name: bucket,
            }
            row.update(vals)
            rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["mean_k"] = safe_div(df["sum_k"], df["n_blocks"])
        df["token_acceptance_rate"] = safe_div(df["accepted_tokens"], df["draft_tokens"])
        df["token_acceptance_pct"] = 100 * df["token_acceptance_rate"]
        df["wasted_token_rate"] = safe_div(df["rejected_tokens"], df["draft_tokens"])
        df["wasted_token_pct"] = 100 * df["wasted_token_rate"]
        return df

    entropy_df = counter_to_df(entropy_counter, "entropy_bin")
    repetition_df = counter_to_df(repetition_counter, "repetition_bin")

    feat_rows = []
    for key, vals in feature_acc.items():
        run_name, pool, method, workload, model_variant, profile, candidate_set, feature = key
        n = vals["n"]
        feat_rows.append({
            "run_name": run_name,
            "pool": pool,
            "method": method,
            "workload": workload,
            "model_variant": model_variant,
            "profile": profile,
            "candidate_set": candidate_set,
            "feature": feature,
            "n": n,
            "mean_value": vals["sum_value"] / n if n else np.nan,
        })
    feature_df = pd.DataFrame(feat_rows)

    return k_df, kw_df, entropy_df, repetition_df, feature_df


# -----------------------------
# Fixed entropy/repetition tables
# -----------------------------
def fixed_entropy_repetition_tables(fixed_req: pd.DataFrame):
    if fixed_req.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x = fixed_req.copy()
    if "workload" in x.columns:
        x["workload"] = x["workload"].map(normalize_workload)

    group_base = [c for c in ["method", "k", "baseline_config", "workload"] if c in x.columns]

    entropy_tables = []
    if "entropy_bucket" in x.columns:
        t = aggregate_request_rows(x, group_base + ["entropy_bucket"])
        entropy_tables.append(t)

    if "mean_entropy" in x.columns:
        x["mean_entropy"] = to_num(x["mean_entropy"])
        x["entropy_bin"] = x["mean_entropy"].map(bucket_entropy)
        t = aggregate_request_rows(x, group_base + ["entropy_bin"])
        entropy_tables.append(t)

    fixed_entropy = pd.concat(entropy_tables, ignore_index=True) if entropy_tables else pd.DataFrame()

    fixed_rep = pd.DataFrame()
    if "repetition_density" in x.columns:
        x["repetition_density"] = to_num(x["repetition_density"])
        x["repetition_bin"] = x["repetition_density"].map(bucket_repetition)
        fixed_rep = aggregate_request_rows(x, group_base + ["repetition_bin"])

    useful_cols = [
        "mean_entropy",
        "repetition_density",
        "acceptance_rate",
        "rejection_rate",
        "first_rejection_pos",
        "acceptance_decay_slope",
        "rejection_concentration",
        "rejection_severity",
        "entropy_range",
        "high_entropy_frac",
        "accepted_per_draft",
        "accepted_tokens_per_verifier_pass",
        "prefix_cache_hit_rate",
        "kv_cache_usage_perc",
    ]
    useful_cols = [c for c in useful_cols if c in x.columns]

    if useful_cols:
        agg = {f"mean_{c}": (c, "mean") for c in useful_cols}
        for c in useful_cols:
            x[c] = to_num(x[c])
        fixed_features = x.groupby(group_base, dropna=False).agg(**agg).reset_index()
    else:
        fixed_features = pd.DataFrame()

    return fixed_entropy, fixed_rep, fixed_features


# -----------------------------
# Pareto / top config tables
# -----------------------------
def pareto_frontier(df: pd.DataFrame):
    if df.empty or "mean_tps" not in df.columns or "wasted_token_pct" not in df.columns:
        return pd.DataFrame()

    x = df.copy()
    x = x[x.get("pool", "slow_fast").astype(str).eq("slow_fast")] if "pool" in x.columns else x
    x = x.dropna(subset=["mean_tps", "wasted_token_pct"])
    if x.empty:
        return x

    # Non-dominated: no other row has >= TPS and <= waste, with one strict.
    rows = []
    vals = x[["mean_tps", "wasted_token_pct"]].to_numpy()
    for i, row in enumerate(x.itertuples(index=False)):
        tps_i, waste_i = vals[i]
        dominated = False
        for j, (tps_j, waste_j) in enumerate(vals):
            if j == i:
                continue
            if (tps_j >= tps_i and waste_j <= waste_i) and (tps_j > tps_i or waste_j < waste_i):
                dominated = True
                break
        if not dominated:
            rows.append(i)

    return x.iloc[rows].sort_values(["wasted_token_pct", "mean_tps"], ascending=[True, False])


def top_configs(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()

    x = df.copy()
    if "pool" in x.columns:
        x = x[x["pool"].astype(str).eq("slow_fast")].copy()

    metrics = [
        ("highest_tps", "mean_tps", False),
        ("highest_speedup_vs_ar", "speedup_vs_ar", False),
        ("lowest_waste_pct", "wasted_token_pct", True),
        ("highest_acceptance_pct", "token_acceptance_pct", False),
        ("highest_STE", "STE_speed_adjusted_token_efficiency", False),
        ("highest_ATE", "ATE_accepted_throughput_efficiency", False),
    ]

    rows = []
    for label, col, ascending in metrics:
        if col not in x.columns:
            continue
        tmp = x.dropna(subset=[col]).sort_values(col, ascending=ascending).head(10).copy()
        tmp.insert(0, "ranking_metric", label)
        tmp.insert(1, "ranking_value", tmp[col])
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def apex_vs_slow_only(apex_all: pd.DataFrame):
    if apex_all.empty or "pool" not in apex_all.columns:
        return pd.DataFrame()

    keys = [c for c in ["run_name", "model_variant", "profile", "candidate_set"] if c in apex_all.columns]
    sf = apex_all[apex_all["pool"].astype(str).eq("slow_fast")].copy()
    so = apex_all[apex_all["pool"].astype(str).eq("slow_only")].copy()

    if sf.empty or so.empty:
        return pd.DataFrame()

    keep = keys + [
        c for c in [
            "mean_tps",
            "token_acceptance_pct",
            "wasted_token_pct",
            "accepted_per_block",
            "mean_k",
            "full_accept_rate",
            "speedup_vs_ar",
            "STE_speed_adjusted_token_efficiency",
            "ATE_accepted_throughput_efficiency",
        ]
        if c in apex_all.columns
    ]

    sf = sf[keep].copy()
    so = so[keep].copy()

    out = sf.merge(so, on=keys, how="inner", suffixes=("_slow_fast", "_slow_only"))

    if "mean_tps_slow_fast" in out.columns and "mean_tps_slow_only" in out.columns:
        out["tps_ratio_slow_fast_vs_slow_only"] = safe_div(out["mean_tps_slow_fast"], out["mean_tps_slow_only"])

    if "wasted_token_pct_slow_fast" in out.columns and "wasted_token_pct_slow_only" in out.columns:
        out["waste_pct_delta_slow_fast_minus_slow_only"] = (
            out["wasted_token_pct_slow_fast"] - out["wasted_token_pct_slow_only"]
        )

    if "token_acceptance_pct_slow_fast" in out.columns and "token_acceptance_pct_slow_only" in out.columns:
        out["acceptance_pct_delta_slow_fast_minus_slow_only"] = (
            out["token_acceptance_pct_slow_fast"] - out["token_acceptance_pct_slow_only"]
        )

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-root",
        default="results/apexp_block/config_large_500/evaluation/paper_cap50_profile_kgrid",
        help="Root containing fixed_runs, fixed_baseline_summary, apex_hot_runs, apex_hot_summary.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <eval-root>/paper_tables",
    )
    ap.add_argument(
        "--skip-raw-apex-scan",
        action="store_true",
        help="Skip raw block_events scan. Faster, but no k-by-workload/entropy/repetition tables from APEX blocks.",
    )
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir) if args.out_dir else eval_root / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_root] {eval_root}")
    print(f"[out_dir]   {out_dir}")

    warnings = []

    manifest = load_manifest(eval_root)

    # Fixed baselines
    fixed_req = load_fixed_request_rows(eval_root)
    if fixed_req.empty:
        warnings.append("No fixed baseline request rows found.")
    else:
        fixed_req.to_csv(out_dir / "00_fixed_request_rows_raw.csv", index=False)

    ar_tps, workload_ar = get_ar_references(fixed_req)
    if pd.isna(ar_tps):
        warnings.append(
            "Could not find AR baseline rows. speedup_vs_ar will be NaN unless AR is present in fixed_baseline_request_rows.csv."
        )
    else:
        print(f"[AR] overall mean TPS = {ar_tps:.4f}")
        pd.DataFrame(
            [{"scope": "overall", "workload": "ALL", "ar_tps": ar_tps}]
            + [{"scope": "workload", "workload": k, "ar_tps": v} for k, v in workload_ar.items()]
        ).to_csv(out_dir / "00_ar_reference_tps.csv", index=False)

    fixed_all = aggregate_request_rows(
        fixed_req,
        [c for c in ["method", "k", "baseline_config"] if c in fixed_req.columns],
        ar_tps=ar_tps,
        workload_ar=None,
    )
    fixed_all.to_csv(out_dir / "02_fixed_all_configs_overall.csv", index=False)

    fixed_by_workload = aggregate_request_rows(
        fixed_req,
        [c for c in ["method", "k", "baseline_config", "workload"] if c in fixed_req.columns],
        ar_tps=ar_tps,
        workload_ar=workload_ar,
    )
    fixed_by_workload.to_csv(out_dir / "04_fixed_all_configs_by_workload.csv", index=False)

    fixed_by_baseline = aggregate_request_rows(
        fixed_req,
        [c for c in ["method"] if c in fixed_req.columns],
        ar_tps=ar_tps,
        workload_ar=None,
    )
    fixed_by_baseline.to_csv(out_dir / "05_fixed_grouped_by_baseline_method.csv", index=False)

    fixed_entropy, fixed_rep, fixed_features = fixed_entropy_repetition_tables(fixed_req)
    fixed_entropy.to_csv(out_dir / "08_fixed_entropy_distribution.csv", index=False)
    fixed_rep.to_csv(out_dir / "09_fixed_repetition_distribution.csv", index=False)
    fixed_features.to_csv(out_dir / "10_fixed_useful_feature_summary.csv", index=False)

    # APEX summaries
    req_overall, block_overall, req_workload, block_workload, kdist_summary = load_apex_summary_tables(
        eval_root, manifest
    )

    apex_all = merge_apex_request_block(req_overall, block_overall, keys=["run_name", "pool"])
    apex_all = add_apex_metadata(apex_all, manifest)
    apex_all = add_speedups(apex_all, ar_tps, workload_ar)
    apex_all.to_csv(out_dir / "01_apex_all_configs_overall_all_pools.csv", index=False)

    apex_slow_fast = apex_all.copy()
    if "pool" in apex_slow_fast.columns:
        apex_slow_fast = apex_slow_fast[apex_slow_fast["pool"].astype(str).eq("slow_fast")].copy()
    apex_slow_fast.to_csv(out_dir / "01_apex_all_configs_overall_slow_fast_only.csv", index=False)

    apex_by_workload = merge_apex_request_block(
        req_workload,
        block_workload,
        keys=["run_name", "pool", "method", "workload"],
    )
    apex_by_workload = add_apex_metadata(apex_by_workload, manifest)
    apex_by_workload = add_speedups(apex_by_workload, ar_tps, workload_ar)
    apex_by_workload.to_csv(out_dir / "03_apex_all_configs_by_workload_all_pools.csv", index=False)

    apex_by_workload_sf = apex_by_workload.copy()
    if "pool" in apex_by_workload_sf.columns:
        apex_by_workload_sf = apex_by_workload_sf[apex_by_workload_sf["pool"].astype(str).eq("slow_fast")].copy()
    apex_by_workload_sf.to_csv(out_dir / "03_apex_all_configs_by_workload_slow_fast_only.csv", index=False)

    # Existing k distribution summary.
    if not kdist_summary.empty:
        if "n" in kdist_summary.columns and "n_blocks" not in kdist_summary.columns:
            kdist_summary["n_blocks"] = kdist_summary["n"]

        kdist_summary = add_apex_metadata(kdist_summary, manifest)

        group_cols = [c for c in ["run_name", "pool", "method"] if c in kdist_summary.columns]
        if "n_blocks" in kdist_summary.columns and group_cols:
            kdist_summary["pct_within_run_pool_method"] = (
                kdist_summary["n_blocks"] / kdist_summary.groupby(group_cols)["n_blocks"].transform("sum")
            )

        kdist_summary.to_csv(out_dir / "06_apex_k_distribution_from_summary.csv", index=False)

    # Raw APEX scan for workload-level k and entropy/repetition.
    if not args.skip_raw_apex_scan:
        k_raw, k_work_raw, apex_entropy, apex_rep, apex_feature_summary = scan_apex_raw(eval_root, manifest)
        k_raw.to_csv(out_dir / "06_apex_k_distribution_overall_from_block_events.csv", index=False)
        k_work_raw.to_csv(out_dir / "07_apex_k_distribution_by_workload_from_block_events.csv", index=False)
        apex_entropy.to_csv(out_dir / "11_apex_entropy_distribution_from_block_events.csv", index=False)
        apex_rep.to_csv(out_dir / "12_apex_repetition_distribution_from_block_events.csv", index=False)
        apex_feature_summary.to_csv(out_dir / "13_apex_useful_feature_summary_from_block_events.csv", index=False)

    # Extra useful tables
    delta = apex_vs_slow_only(apex_all)
    delta.to_csv(out_dir / "14_apex_vs_slow_only_delta.csv", index=False)

    pf = pareto_frontier(apex_slow_fast)
    pf.to_csv(out_dir / "15_apex_pareto_frontier_tps_vs_waste.csv", index=False)

    top = top_configs(apex_slow_fast)
    top.to_csv(out_dir / "16_apex_top_configs_for_paper.csv", index=False)

    fixed_pf = pareto_frontier(fixed_all)
    fixed_pf.to_csv(out_dir / "17_fixed_pareto_frontier_tps_vs_waste.csv", index=False)

    # Best fixed vs best APEX compact table
    comp_rows = []
    if not apex_slow_fast.empty:
        for metric, ascending in [
            ("mean_tps", False),
            ("wasted_token_pct", True),
            ("token_acceptance_pct", False),
            ("STE_speed_adjusted_token_efficiency", False),
            ("ATE_accepted_throughput_efficiency", False),
        ]:
            if metric in apex_slow_fast.columns:
                row = apex_slow_fast.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(1)
                if not row.empty:
                    r = row.iloc[0].to_dict()
                    r["system_family"] = "APEX"
                    r["selection_metric"] = metric
                    comp_rows.append(r)

    if not fixed_all.empty:
        for metric, ascending in [
            ("mean_tps", False),
            ("wasted_token_pct", True),
            ("token_acceptance_pct", False),
            ("STE_speed_adjusted_token_efficiency", False),
            ("ATE_accepted_throughput_efficiency", False),
        ]:
            if metric in fixed_all.columns:
                row = fixed_all.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(1)
                if not row.empty:
                    r = row.iloc[0].to_dict()
                    r["system_family"] = "fixed_baseline"
                    r["selection_metric"] = metric
                    comp_rows.append(r)

    pd.DataFrame(comp_rows).to_csv(out_dir / "18_best_apex_vs_best_fixed_by_metric.csv", index=False)

    # Inventory
    inv = {
        "eval_root": str(eval_root),
        "out_dir": str(out_dir),
        "n_fixed_request_rows": len(fixed_req),
        "n_apex_overall_rows_all_pools": len(apex_all),
        "n_apex_overall_rows_slow_fast": len(apex_slow_fast),
        "n_apex_workload_rows_all_pools": len(apex_by_workload),
        "n_apex_workload_rows_slow_fast": len(apex_by_workload_sf),
        "ar_tps_reference": ar_tps if pd.notna(ar_tps) else None,
        "n_ar_workload_refs": len(workload_ar),
    }
    pd.DataFrame([inv]).to_csv(out_dir / "00_table_build_inventory.csv", index=False)

    # Write warnings
    (out_dir / "WARNINGS.txt").write_text("\n".join(warnings) + ("\n" if warnings else "No warnings.\n"))

    # Optional Excel bundle
    xlsx_path = out_dir / "paper_tables.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            files = [
                "00_table_build_inventory.csv",
                "00_ar_reference_tps.csv",
                "01_apex_all_configs_overall_slow_fast_only.csv",
                "01_apex_all_configs_overall_all_pools.csv",
                "02_fixed_all_configs_overall.csv",
                "03_apex_all_configs_by_workload_slow_fast_only.csv",
                "03_apex_all_configs_by_workload_all_pools.csv",
                "04_fixed_all_configs_by_workload.csv",
                "05_fixed_grouped_by_baseline_method.csv",
                "06_apex_k_distribution_from_summary.csv",
                "06_apex_k_distribution_overall_from_block_events.csv",
                "07_apex_k_distribution_by_workload_from_block_events.csv",
                "08_fixed_entropy_distribution.csv",
                "09_fixed_repetition_distribution.csv",
                "10_fixed_useful_feature_summary.csv",
                "11_apex_entropy_distribution_from_block_events.csv",
                "12_apex_repetition_distribution_from_block_events.csv",
                "13_apex_useful_feature_summary_from_block_events.csv",
                "14_apex_vs_slow_only_delta.csv",
                "15_apex_pareto_frontier_tps_vs_waste.csv",
                "16_apex_top_configs_for_paper.csv",
                "17_fixed_pareto_frontier_tps_vs_waste.csv",
                "18_best_apex_vs_best_fixed_by_metric.csv",
            ]
            for fname in files:
                p = out_dir / fname
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                sheet = re.sub(r"[^A-Za-z0-9_]", "_", fname.replace(".csv", ""))[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"[wrote] {xlsx_path}")
    except Exception as e:
        print(f"[warn] Excel workbook not written: {e}")

    print("\n[DONE] tables written to:")
    print(out_dir)
    print("\nKey tables:")
    for x in [
        "01_apex_all_configs_overall_slow_fast_only.csv",
        "02_fixed_all_configs_overall.csv",
        "03_apex_all_configs_by_workload_slow_fast_only.csv",
        "04_fixed_all_configs_by_workload.csv",
        "05_fixed_grouped_by_baseline_method.csv",
        "07_apex_k_distribution_by_workload_from_block_events.csv",
        "11_apex_entropy_distribution_from_block_events.csv",
        "12_apex_repetition_distribution_from_block_events.csv",
        "16_apex_top_configs_for_paper.csv",
        "18_best_apex_vs_best_fixed_by_metric.csv",
    ]:
        print("  ", out_dir / x)


if __name__ == "__main__":
    main()
