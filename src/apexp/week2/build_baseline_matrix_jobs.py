from __future__ import annotations

import argparse
import csv
from pathlib import Path


INPUTS = [
    ("mathematical_reasoning", "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl", 512),
    ("long_horizon_swe", "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl", 1024),
    ("long_context_completion", "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_context_completion_synthetic.jsonl", 2048),
    ("long_chain_reasoning", "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/long_chain_reasoning_synthetic.jsonl", 1024),
    ("conversational_generation_sft", "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl", 512),
    ("conversational_generation_gen", "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl", 512),
    ("code_gen", "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/code_gen_train.jsonl", 512),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--k-values", default="1,2,4,8,16")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    rows = []
    job_i = 0

    for workload, input_file, max_tokens in INPUTS:
        rows.append({
            "job_id": f"job{job_i:05d}",
            "priority": 0,
            "workload": workload,
            "input_file": input_file,
            "method": "ar",
            "k": "NA",
            "temperature": args.temperature,
            "limit": args.limit,
            "max_tokens": max_tokens,
            "run_dir": str(out_root / "runs" / workload / "ar" / "kNA"),
        })
        job_i += 1

        for method in ["ngram_sd", "draft_sd", "eagle3"]:
            for k in k_values:
                rows.append({
                    "job_id": f"job{job_i:05d}",
                    "priority": 1,
                    "workload": workload,
                    "input_file": input_file,
                    "method": method,
                    "k": k,
                    "temperature": args.temperature,
                    "limit": args.limit,
                    "max_tokens": max_tokens,
                    "run_dir": str(out_root / "runs" / workload / method / f"k{k}"),
                })
                job_i += 1

    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "job_id", "priority", "workload", "input_file", "method", "k",
        "temperature", "limit", "max_tokens", "run_dir",
    ]

    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {out_tsv}")
    print(f"jobs: {len(rows)}")
    print(f"limit per workload/config: {args.limit}")


if __name__ == "__main__":
    main()
