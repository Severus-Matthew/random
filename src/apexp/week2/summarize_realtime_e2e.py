from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd


def load_result_files(out_dir: Path) -> pd.DataFrame:
    rows = []
    for p in (out_dir / "results").glob("*.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_tps(run_dir: str) -> dict:
    p = Path(run_dir) / "summary.csv"
    if not p.exists():
        return {"has_summary": False, "tps": None, "tps_col": None}

    df = pd.read_csv(p)
    if len(df) == 0:
        return {"has_summary": False, "tps": None, "tps_col": None}

    row = df.iloc[0].to_dict()
    out = {"has_summary": True}

    for c in ["tokens_per_sec", "target_tokens_per_sec", "actual_tps", "throughput_tps", "tps"]:
        if c in row and pd.notna(row[c]):
            out["tps"] = float(row[c])
            out["tps_col"] = c
            break

    for c in ["latency_s", "wall_time_s", "elapsed_s"]:
        if c in row and pd.notna(row[c]):
            out["latency_s"] = float(row[c])
            break

    for c in ["num_output_tokens", "output_tokens", "generated_tokens", "tokens"]:
        if c in row and pd.notna(row[c]):
            out["output_tokens"] = float(row[c])
            break

    return out


def k_distribution(root: Path) -> dict:
    c = Counter()
    for p in root.rglob("block_events.jsonl"):
        for line in open(p):
            try:
                e = json.loads(line)
                c[int(e.get("k_actual", -1))] += 1
            except Exception:
                pass
    return dict(sorted(c.items()))


def k_distribution_by_run(out_dir: Path) -> pd.DataFrame:
    rows = []
    fast_root = out_dir / "slow_fast"
    for run_dir in sorted(fast_root.rglob("run")):
        c = Counter()
        for p in run_dir.rglob("block_events.jsonl"):
            for line in open(p):
                try:
                    e = json.loads(line)
                    c[int(e.get("k_actual", -1))] += 1
                except Exception:
                    pass
        if c:
            # parent structure: slow_fast/method/task_id/run
            method = run_dir.parents[1].name if len(run_dir.parents) > 1 else "unknown"
            task_id = run_dir.parent.name
            row = {"task_id": task_id, "method": method}
            for k, v in sorted(c.items()):
                row[f"k_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def safe_ratio(a, b):
    if pd.isna(a) or pd.isna(b) or b <= 0:
        return None
    return float(a) / float(b)


def summarize_group(g: pd.DataFrame, group_name: dict) -> dict:
    row = dict(group_name)
    row["n"] = int(len(g))
    row["slow_mean_tps"] = float(g["tps_slow"].mean())
    row["fast_mean_tps"] = float(g["tps_fast"].mean())
    row["fast_over_slow_ratio_of_means"] = safe_ratio(row["fast_mean_tps"], row["slow_mean_tps"])
    row["fast_minus_slow_tps"] = float(g["tps_fast"].mean() - g["tps_slow"].mean())
    row["mean_paired_fast_over_slow"] = float((g["tps_fast"] / g["tps_slow"]).mean())
    row["median_paired_fast_over_slow"] = float((g["tps_fast"] / g["tps_slow"]).median())

    if "ar_tps" in g.columns:
        valid = g[g["ar_tps"].notna() & (g["ar_tps"] > 0)].copy()
        row["n_with_ar"] = int(len(valid))
        if len(valid):
            row["ar_mean_tps"] = float(valid["ar_tps"].mean())
            row["slow_over_ar_ratio_of_means"] = safe_ratio(valid["tps_slow"].mean(), valid["ar_tps"].mean())
            row["fast_over_ar_ratio_of_means"] = safe_ratio(valid["tps_fast"].mean(), valid["ar_tps"].mean())
            row["mean_paired_slow_over_ar"] = float((valid["tps_slow"] / valid["ar_tps"]).mean())
            row["mean_paired_fast_over_ar"] = float((valid["tps_fast"] / valid["ar_tps"]).mean())

    return row


def load_ar_baseline(path: str | None, tps_col: str | None) -> pd.DataFrame | None:
    if not path:
        return None

    p = Path(path)
    df = pd.read_csv(p)

    if tps_col is None:
        for c in ["tokens_per_sec", "target_tokens_per_sec", "actual_tps", "throughput_tps", "tps", "ar_tps"]:
            if c in df.columns:
                tps_col = c
                break

    if tps_col is None or tps_col not in df.columns:
        raise SystemExit(f"Could not find AR TPS column in {p}. Columns: {list(df.columns)}")

    # Standardize join columns.
    if "ar_tps" not in df.columns:
        df["ar_tps"] = df[tps_col]

    keep = [c for c in ["prompt_hash", "source_file", "source_index", "workload", "ar_tps"] if c in df.columns]
    if "prompt_hash" not in keep:
        raise SystemExit("AR baseline must contain prompt_hash for clean joining.")

    return df[keep].drop_duplicates(subset=["prompt_hash"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ar-baseline-csv", default="")
    ap.add_argument("--ar-tps-col", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    df = load_result_files(out_dir)
    if len(df) == 0:
        raise SystemExit(f"No result files found under {out_dir}/results")

    metric_rows = []
    for _, r in df.iterrows():
        m = load_tps(r["run_dir"])
        metric_rows.append({**r.to_dict(), **m})

    mdf = pd.DataFrame(metric_rows)
    mdf.to_csv(out_dir / "request_metrics.csv", index=False)

    slow = mdf[mdf["pool"] == "slow_only"].copy()
    fast = mdf[mdf["pool"] == "slow_fast"].copy()

    joined = slow.merge(
        fast,
        on="global_index",
        suffixes=("_slow", "_fast"),
    )

    # Normalize common columns after merge.
    joined["prompt_hash"] = joined["prompt_hash_slow"]
    joined["source_file"] = joined["source_file_slow"]
    joined["source_index"] = joined["source_index_slow"]
    joined["workload"] = joined["workload_slow"]
    joined["method"] = joined["method_slow"]
    joined["slow_k"] = joined["slow_k_slow"]

    joined = joined[
        joined["tps_slow"].notna()
        & joined["tps_fast"].notna()
        & (joined["tps_slow"] > 0)
        & (joined["tps_fast"] > 0)
    ].copy()

    joined["fast_over_slow"] = joined["tps_fast"] / joined["tps_slow"]
    joined["fast_minus_slow_tps"] = joined["tps_fast"] - joined["tps_slow"]

    ar = load_ar_baseline(args.ar_baseline_csv or None, args.ar_tps_col or None)
    if ar is not None:
        joined = joined.merge(ar[["prompt_hash", "ar_tps"]], on="prompt_hash", how="left")
        joined["slow_over_ar"] = joined["tps_slow"] / joined["ar_tps"]
        joined["fast_over_ar"] = joined["tps_fast"] / joined["ar_tps"]

    joined.to_csv(out_dir / "request_compare.csv", index=False)

    # Overall summary.
    overall = summarize_group(joined, {"group": "overall"})
    overall["n_slow_results"] = int(len(slow))
    overall["n_fast_results"] = int(len(fast))
    overall["slow_success"] = int((slow["status"] == "SUCCESS").sum())
    overall["fast_success"] = int((fast["status"] == "SUCCESS").sum())
    overall["fast_k_distribution"] = json.dumps(k_distribution(out_dir / "slow_fast"), sort_keys=True)

    pd.DataFrame([overall]).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2, sort_keys=True))

    # Group by workload.
    workload_rows = []
    for workload, g in joined.groupby("workload"):
        workload_rows.append(summarize_group(g, {"workload": workload}))
    pd.DataFrame(workload_rows).sort_values("workload").to_csv(
        out_dir / "workload_summary.csv", index=False
    )

    # Group by workload + method + slow_k.
    action_rows = []
    for keys, g in joined.groupby(["workload", "method", "slow_k"]):
        workload, method, slow_k = keys
        action_rows.append(summarize_group(g, {
            "workload": workload,
            "method": method,
            "slow_k": slow_k,
        }))
    pd.DataFrame(action_rows).sort_values(["workload", "method", "slow_k"]).to_csv(
        out_dir / "action_summary.csv", index=False
    )

    # K distribution details.
    kdf = k_distribution_by_run(out_dir)
    if len(kdf):
        kdf.to_csv(out_dir / "fast_k_distribution_by_request.csv", index=False)

    print(json.dumps(overall, indent=2, sort_keys=True))
    print("wrote", out_dir / "request_metrics.csv")
    print("wrote", out_dir / "request_compare.csv")
    print("wrote", out_dir / "summary.csv")
    print("wrote", out_dir / "workload_summary.csv")
    print("wrote", out_dir / "action_summary.csv")
    if len(kdf):
        print("wrote", out_dir / "fast_k_distribution_by_request.csv")


if __name__ == "__main__":
    main()
