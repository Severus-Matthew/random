from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_hot_results(out_dir: Path) -> pd.DataFrame:
    rows = []
    for p in (out_dir / "results").glob("*.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    if not rows:
        raise SystemExit(f"No results under {out_dir}/results")
    return pd.DataFrame(rows)


def normalize_baseline(path: str, name: str, tps_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if tps_col is None:
        for c in ["tokens_per_sec", "target_tokens_per_sec", "actual_tps", "throughput_tps", "tps"]:
            if c in df.columns:
                tps_col = c
                break
    if tps_col is None or tps_col not in df.columns:
        raise SystemExit(f"Could not find TPS col in {path}; columns={list(df.columns)}")

    if "prompt_hash" not in df.columns:
        raise SystemExit(f"{path} must contain prompt_hash")

    keep = ["prompt_hash"]
    for c in ["source_file", "source_index", "workload"]:
        if c in df.columns:
            keep.append(c)

    out = df[keep].copy()
    out[f"{name}_tps"] = df[tps_col].astype(float)
    out = out.drop_duplicates(subset=["prompt_hash"])
    return out


def paired_bootstrap(x, y, n_boot=2000, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n == 0:
        return {}

    rng = np.random.default_rng(seed)
    diffs = []
    ratios = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        diffs.append(float(np.mean(yb - xb)))
        ratios.append(float(np.mean(yb) / np.mean(xb)))

    return {
        "n": int(n),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "mean_diff_y_minus_x": float(np.mean(y - x)),
        "ratio_of_means_y_over_x": float(np.mean(y) / np.mean(x)),
        "diff_ci_low": float(np.percentile(diffs, 2.5)),
        "diff_ci_high": float(np.percentile(diffs, 97.5)),
        "ratio_ci_low": float(np.percentile(ratios, 2.5)),
        "ratio_ci_high": float(np.percentile(ratios, 97.5)),
    }


def summarize(df: pd.DataFrame, group_cols: list[str], baseline_cols: list[str], n_boot: int, seed: int):
    rows = []

    groups = [(("overall",), df)] if not group_cols else df.groupby(group_cols, dropna=False)

    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)

        row = {}
        if group_cols:
            for c, v in zip(group_cols, key):
                row[c] = v
        else:
            row["group"] = "overall"

        row["n"] = int(len(g))
        row["slow_mean_tps"] = float(g["tokens_per_sec_slow"].mean())
        row["fast_mean_tps"] = float(g["tokens_per_sec_fast"].mean())
        row["fast_over_slow"] = float(g["tokens_per_sec_fast"].mean() / g["tokens_per_sec_slow"].mean())

        bs = paired_bootstrap(
            g["tokens_per_sec_slow"],
            g["tokens_per_sec_fast"],
            n_boot=n_boot,
            seed=seed,
        )
        for k, v in bs.items():
            row[f"fast_vs_slow_{k}"] = v

        for b in baseline_cols:
            bcol = f"{b}_tps"
            if bcol in g.columns:
                valid = g[g[bcol].notna() & (g[bcol] > 0)]
                row[f"n_with_{b}"] = int(len(valid))
                if len(valid):
                    row[f"slow_over_{b}"] = float(valid["tokens_per_sec_slow"].mean() / valid[bcol].mean())
                    row[f"fast_over_{b}"] = float(valid["tokens_per_sec_fast"].mean() / valid[bcol].mean())

                    bs_s = paired_bootstrap(valid[bcol], valid["tokens_per_sec_slow"], n_boot=n_boot, seed=seed)
                    bs_f = paired_bootstrap(valid[bcol], valid["tokens_per_sec_fast"], n_boot=n_boot, seed=seed)
                    row[f"slow_vs_{b}_ratio_ci_low"] = bs_s.get("ratio_ci_low")
                    row[f"slow_vs_{b}_ratio_ci_high"] = bs_s.get("ratio_ci_high")
                    row[f"fast_vs_{b}_ratio_ci_low"] = bs_f.get("ratio_ci_low")
                    row[f"fast_vs_{b}_ratio_ci_high"] = bs_f.get("ratio_ci_high")

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot-out-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline", action="append", default=[],
                    help="name:path:tps_col . Example ar:/path/ar.csv:tokens_per_sec")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hot = load_hot_results(Path(args.hot_out_dir))

    slow = hot[hot["pool"] == "slow_only"].copy()
    fast = hot[hot["pool"] == "slow_fast"].copy()

    joined = slow.merge(
        fast,
        on="global_index",
        suffixes=("_slow", "_fast"),
    )

    joined["prompt_hash"] = joined["prompt_hash_slow"]
    joined["workload"] = joined["workload_slow"]
    joined["method"] = joined["method_slow"]
    joined["slow_k"] = joined["slow_k_slow"]
    joined["source_file"] = joined["source_file_slow"]
    joined["source_index"] = joined["source_index_slow"]

    joined = joined[
        joined["tokens_per_sec_slow"].notna()
        & joined["tokens_per_sec_fast"].notna()
        & (joined["tokens_per_sec_slow"] > 0)
        & (joined["tokens_per_sec_fast"] > 0)
    ].copy()

    baseline_names = []
    for spec in args.baseline:
        parts = spec.split(":")
        if len(parts) < 2:
            raise SystemExit("--baseline must be name:path:tps_col")
        name = parts[0]
        path = parts[1]
        tps_col = parts[2] if len(parts) >= 3 and parts[2] else None
        bdf = normalize_baseline(path, name, tps_col)
        joined = joined.merge(bdf[["prompt_hash", f"{name}_tps"]], on="prompt_hash", how="left")
        baseline_names.append(name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joined.to_csv(out_dir / "joined_request_metrics.csv", index=False)

    summarize(joined, [], baseline_names, args.n_boot, args.seed).to_csv(
        out_dir / "summary_overall.csv", index=False
    )
    summarize(joined, ["workload"], baseline_names, args.n_boot, args.seed).to_csv(
        out_dir / "summary_by_workload.csv", index=False
    )
    summarize(joined, ["workload", "method", "slow_k"], baseline_names, args.n_boot, args.seed).to_csv(
        out_dir / "summary_by_workload_method_k.csv", index=False
    )

    print("wrote", out_dir / "joined_request_metrics.csv")
    print("wrote", out_dir / "summary_overall.csv")
    print("wrote", out_dir / "summary_by_workload.csv")
    print("wrote", out_dir / "summary_by_workload_method_k.csv")


if __name__ == "__main__":
    main()
