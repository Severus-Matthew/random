from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot-out-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    root = Path(args.hot_out_dir)
    rows = []

    for p in root.rglob("p*/trace/block_events.jsonl"):
        for e in read_jsonl(p):
            gen = e.get("generated_token_ids") or []
            acc = int(e.get("accepted_len") or 0)

            accepted_token_ids = gen[:acc]
            correction_token_ids = gen[acc:]

            rows.append({
                "task_id": e.get("task_id"),
                "pool": e.get("pool"),
                "mode": e.get("mode"),
                "workload": e.get("workload"),
                "method": e.get("method"),
                "prompt_hash": e.get("prompt_hash"),
                "global_index": e.get("global_index"),
                "block_index": e.get("block_index"),
                "k_requested": e.get("k_requested"),
                "slow_k": e.get("slow_k"),
                "k_actual": e.get("k_actual"),
                "accepted_len": e.get("accepted_len"),
                "num_rejected": e.get("num_rejected"),
                "first_rejection": e.get("first_rejection"),
                "full_accept": e.get("full_accept"),
                "accepted_mask": json.dumps(e.get("accepted_mask")),
                "generated_token_ids": json.dumps(gen),
                "accepted_token_ids_from_generated_prefix": json.dumps(accepted_token_ids),
                "correction_or_new_token_ids": json.dumps(correction_token_ids),
                "draft_token_ids_available": bool(e.get("draft_token_ids")),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("wrote", args.out_csv)
    print("rows:", len(df))
    print("by pool/mode:")
    print(df.groupby(["pool", "mode"]).size())
    print("\ndynamic k distribution:")
    print(
        df[df["mode"] == "dynamic"]
        .groupby(["workload", "method", "k_actual", "accepted_len"])
        .size()
        .sort_values(ascending=False)
        .head(60)
    )


if __name__ == "__main__":
    main()
