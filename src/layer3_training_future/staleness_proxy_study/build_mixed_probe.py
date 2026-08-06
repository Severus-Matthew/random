#!/usr/bin/env python
"""
build_mixed_probe.py — balanced cross-workload probe for output-KL drift
========================================================================
The output-KL drift distance (measure_output_kl.py) should reflect the SAME
mix of text the drafter actually faces, not one domain — RLHF/instruct drift
moves reasoning/format tokens far more than boilerplate, so a math-only probe
mis-calibrates the acceptance-vs-KL x-axis on other workloads.

This samples a fixed number of prompts per workload from the Phase-1 test
splits and writes one balanced probe_mixed.jsonl. Each kept row carries a
`workload` field so you can OPTIONALLY also compute per-workload KL later
(by filtering the same file) without re-sampling.

Deterministic (seeded) so the probe is reproducible across families.

Usage:
  python build_mixed_probe.py \
      --data_dir data/benchmarks/By_split_phase_1 --suffix _test.jsonl \
      --per_workload 14 --out data/benchmarks/probe_mixed.jsonl
"""
import argparse
import json
import random
from pathlib import Path

DEFAULT_WORKLOADS = [
    "mathematical_reasoning", "code_gen", "long_chain_reasoning",
    "long_context_completion", "conversational_generation_gen",
    "hardware_gen", "long_horizon_swe",
]

PROMPT_KEYS = ("prompt", "input", "question", "problem_statement", "text", "content")


def get_prompt(row):
    for k in PROMPT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--suffix", default="_test.jsonl")
    ap.add_argument("--workloads", nargs="*", default=DEFAULT_WORKLOADS)
    ap.add_argument("--per_workload", type=int, default=14,
                    help="Prompts sampled per workload (14 x 7 ≈ 96).")
    ap.add_argument("--min_chars", type=int, default=16,
                    help="Skip trivially short prompts.")
    ap.add_argument("--max_chars", type=int, default=8000,
                    help="Skip pathologically long prompts (keeps KL fwd-pass cheap).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    kept, per_wl_counts, missing = [], {}, []

    for wl in args.workloads:
        path = data_dir / f"{wl}{args.suffix}"
        if not path.exists():
            missing.append(str(path))
            continue
        rows = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = get_prompt(r)
            if p and args.min_chars <= len(p) <= args.max_chars:
                rows.append(p)
        rng.shuffle(rows)
        take = rows[: args.per_workload]
        for p in take:
            kept.append({"workload": wl, "prompt": p})
        per_wl_counts[wl] = len(take)
        if len(take) < args.per_workload:
            print(f"  [warn] {wl}: only {len(take)} usable prompts "
                  f"(< {args.per_workload})")

    rng.shuffle(kept)   # interleave workloads so any truncation stays balanced
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(kept)} prompts -> {out}")
    for wl, n in per_wl_counts.items():
        print(f"  {wl:32s} {n}")
    if missing:
        print("  [warn] missing splits:", *missing, sep="\n    ")
    print("\nNext:")
    print(f"  python measure_output_kl.py --ref_model <family ref> \\")
    print(f"      --manifest <family>/drift_manifest.csv \\")
    print(f"      --probe_jsonl {out} --n_probe {len(kept)} --max_len 256 \\")
    print(f"      --out <family>/drift_kl.csv")


if __name__ == "__main__":
    main()
