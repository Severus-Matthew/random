#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_events(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(p.rglob("block_events.jsonl"))
        else:
            files = [p]
        for f in files:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    yield f, e


def event_to_row(path, e):
    k = int(e["k_actual"])
    L = int(e["accepted_len"])
    R = int(e["first_rejection"])

    row = {
        "source_file": str(path),
        "run_id": e.get("run_id"),
        "workload": e.get("workload"),
        "method": e.get("method"),
        "temperature": e.get("temperature"),
        "seed": e.get("seed"),
        "req_id": e.get("req_id"),
        "block_index": e.get("block_index"),
        "k_requested": e.get("k_requested"),
        "k_actual": k,
        "accepted_len": L,
        "first_rejection": R,
        "full_accept": bool(e.get("full_accept")),
        "num_rejected": e.get("num_rejected"),
    }

    # Survival labels:
    # survival_pos_j = 1 if token j was accepted, 0 if rejected/reached and failed.
    # For a block with k_actual = k, all positions are observed under vLLM's
    # accepted-prefix outcome.
    for j in range(k):
        row[f"survival_pos_{j}"] = 1 if j < L else 0
        # hazard target: first rejection at j.
        row[f"hazard_event_pos_{j}"] = 1 if (L < k and j == L) else 0
        row[f"at_risk_pos_{j}"] = 1 if j <= L else 0

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/apexp_block/dataset")
    args = ap.parse_args()

    rows = []
    bad = []

    for path, e in read_events(args.inputs):
        try:
            rows.append(event_to_row(path, e))
        except Exception as ex:
            bad.append({"source_file": str(path), "error": repr(ex), "event": e})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "block_survival_train.csv", index=False)

    # cost placeholder; scheduler hook currently does not log timing.
    # Request-level latency can be joined later by run_id/req_id.
    cost_cols = [
        c for c in [
            "source_file", "run_id", "workload", "method", "temperature",
            "seed", "req_id", "block_index", "k_requested", "k_actual",
            "accepted_len", "first_rejection"
        ] if c in df.columns
    ]
    df[cost_cols].to_csv(out_dir / "block_cost_train.csv", index=False)

    if bad:
        with (out_dir / "bad_block_events.jsonl").open("w") as f:
            for b in bad:
                f.write(json.dumps(b) + "\n")

    diagnostics = {
        "n_rows": len(df),
        "n_bad": len(bad),
        "workloads": sorted(df["workload"].dropna().unique().tolist()) if len(df) else [],
        "methods": sorted(df["method"].dropna().unique().tolist()) if len(df) else [],
        "k_values": sorted(df["k_actual"].dropna().unique().tolist()) if len(df) else [],
        "accepted_len_hist": (
            df["accepted_len"].value_counts().sort_index().to_dict() if len(df) else {}
        ),
    }
    (out_dir / "block_dataset_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )

    print(json.dumps(diagnostics, indent=2))
    print("Wrote:")
    print(" ", out_dir / "block_survival_train.csv")
    print(" ", out_dir / "block_cost_train.csv")


if __name__ == "__main__":
    main()
