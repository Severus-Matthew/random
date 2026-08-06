#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def pos_index(key):
    return int(key.split("_")[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    files = sorted(Path(args.results_dir).rglob("apex_layer0_request.jsonl"))

    total = 0
    bad = 0
    strict = 0
    missing_survival_non_ar = 0

    for p in files:
        with p.open("r") as f:
            for line_no, line in enumerate(f, start=1):
                total += 1

                try:
                    row = json.loads(line)
                except Exception:
                    bad += 1
                    print(f"BAD JSON: {p}:{line_no}")
                    continue

                if row.get("apex_strict_block_trace") is True:
                    strict += 1

                method = str(row.get("method", "")).lower()
                s_keys = sorted(
                    [k for k in row if k.startswith("survival_pos_")],
                    key=pos_index,
                )

                if not s_keys and method not in {"ar", "autoregressive"}:
                    missing_survival_non_ar += 1

                prev = 1.0
                for k in s_keys:
                    s = row.get(k)
                    if s is None:
                        continue
                    if s > prev + 1e-6:
                        bad += 1
                        print(f"MONOTONIC VIOLATION: {p}:{line_no} {k}={s} prev={prev}")
                    prev = s

                pmf_sum = row.get("apex_pmf_prefix_plus_tail")
                if pmf_sum is not None and abs(float(pmf_sum) - 1.0) > 1e-5:
                    bad += 1
                    print(f"PMF SUM BAD: {p}:{line_no} sum={pmf_sum}")

    print(
        f"Layer0 request logs: files={len(files)} rows={total} "
        f"bad={bad} strict_block_rows={strict} "
        f"missing_survival_non_ar={missing_survival_non_ar}"
    )

    if strict == 0:
        print(
            "Note: these are request-aggregate traces from summary.csv/vLLM counters, "
            "not strict per-block first-rejection rows."
        )


if __name__ == "__main__":
    main()
