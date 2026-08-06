from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd


def load_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("summary.csv"):
        try:
            df = pd.read_csv(p)
            df["summary_path"] = str(p)
            rows.append(df)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def find_tps_col(df: pd.DataFrame) -> str:
    for c in [
        "tokens_per_sec",
        "target_tokens_per_sec",
        "actual_tps",
        "throughput_tps",
        "tps",
    ]:
        if c in df.columns:
            return c
    raise SystemExit(f"No TPS column found. Columns: {list(df.columns)}")


def block_k_distribution(root: Path) -> Counter:
    c = Counter()
    for p in root.rglob("block_events.jsonl"):
        with open(p) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    c[int(e.get("k_actual", -1))] += 1
                except Exception:
                    pass
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow-root", required=True)
    ap.add_argument("--fast-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    slow_root = Path(args.slow_root)
    fast_root = Path(args.fast_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    slow = load_summaries(slow_root)
    fast = load_summaries(fast_root)

    print("slow rows:", len(slow))
    print("fast rows:", len(fast))

    if len(slow) == 0 or len(fast) == 0:
        raise SystemExit("Missing summary rows.")

    tps_s = find_tps_col(slow)
    tps_f = find_tps_col(fast)

    summary = {
        "slow_n": int(len(slow)),
        "fast_n": int(len(fast)),
        "slow_mean_tps": float(slow[tps_s].mean()),
        "fast_mean_tps": float(fast[tps_f].mean()),
        "diff_fast_minus_slow": float(fast[tps_f].mean() - slow[tps_s].mean()),
        "slow_tps_col": tps_s,
        "fast_tps_col": tps_f,
        "fast_k_distribution": dict(block_k_distribution(fast_root)),
    }

    pd.DataFrame([summary]).to_csv(out, index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
