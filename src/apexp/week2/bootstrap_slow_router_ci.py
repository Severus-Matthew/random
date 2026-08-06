#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def one_per_prompt(df, score_col):
    idx = df.groupby("prompt_hash")[score_col].idxmax()
    return df.loc[idx].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-dir", required=True)
    ap.add_argument("--baseline-method", default="ngram_sd")
    ap.add_argument("--baseline-k", type=int, default=16)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.router_dir)
    test = pd.read_csv(root / "test_candidates_scored.csv")

    learned = one_per_prompt(test, "pred_tps")
    learned = learned[["prompt_hash", "actual_tps", "method", "k_requested"]].rename(
        columns={"actual_tps": "learned_tps"}
    )

    fixed = test[
        (test["method"] == args.baseline_method)
        & (test["k_requested"] == args.baseline_k)
    ].copy()
    fixed = one_per_prompt(fixed, "actual_tps")
    fixed = fixed[["prompt_hash", "actual_tps"]].rename(columns={"actual_tps": "fixed_tps"})

    oracle = one_per_prompt(test, "actual_tps")
    oracle = oracle[["prompt_hash", "actual_tps"]].rename(columns={"actual_tps": "oracle_tps"})

    m = learned.merge(fixed, on="prompt_hash", how="inner").merge(oracle, on="prompt_hash", how="inner")
    m["diff_vs_fixed"] = m["learned_tps"] - m["fixed_tps"]

    rng = np.random.default_rng(args.seed)
    x = m["diff_vs_fixed"].to_numpy()
    n = len(x)

    vals = []
    for _ in range(args.n_bootstrap):
        vals.append(x[rng.integers(0, n, size=n)].mean())

    lo, hi = np.percentile(vals, [2.5, 97.5])

    print("n prompts:", n)
    print("learned mean TPS:", m["learned_tps"].mean())
    print("fixed mean TPS:", m["fixed_tps"].mean())
    print("oracle mean TPS:", m["oracle_tps"].mean())
    print("mean diff learned - fixed:", m["diff_vs_fixed"].mean())
    print("95% bootstrap CI:", lo, hi)
    print("fraction prompts learned better:", (m["diff_vs_fixed"] > 0).mean())

    m.to_csv(root / "paired_learned_vs_fixed_ngram16.csv", index=False)


if __name__ == "__main__":
    main()
