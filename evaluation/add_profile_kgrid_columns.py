#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    manifest = pd.DataFrame(json.loads(Path(args.manifest).read_text()))

    if manifest.empty:
        raise SystemExit("empty manifest")

    keep_cols = [
        "run_name",
        "profile",
        "candidate_set",
        "candidate_ks",
        "min_k",
        "model_variant",
        "model_dir",
    ]
    manifest = manifest[keep_cols].drop_duplicates("run_name")

    for p in summary_dir.glob("*.csv"):
        df = pd.read_csv(p)
        if "run_name" not in df.columns:
            continue

        # Remove stale/auto-parsed columns so manifest is source of truth.
        for c in ["profile", "candidate_set", "candidate_ks", "min_k", "model_variant", "model_dir"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        out = df.merge(manifest, on="run_name", how="left")
        out_path = p.with_name(p.stem + "_with_profiles.csv")
        out.to_csv(out_path, index=False)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
