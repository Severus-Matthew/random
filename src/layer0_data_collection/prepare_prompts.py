#!/usr/bin/env python
import argparse, json, random
from pathlib import Path


def normalize_prompt(x, field_candidates):
    if isinstance(x, dict):
        for k in field_candidates:
            if k in x and x[k]:
                v = x[k]
                if isinstance(v, list):
                    return "\n".join(
                        [m.get("content", str(m)) if isinstance(m, dict) else str(m)
                         for m in v]
                    )
                return str(v)
        return json.dumps(x)[:6000]
    return str(x)


def load_hf_dataset_all_splits(name, limit_per_split, field_candidates):
    """Load ALL available splits and tag each row with its split name."""
    from datasets import load_dataset, get_dataset_config_names

    # Datasets needing a config name
    CONFIG_MAP = {
        "openai/gsm8k":              "main",
        "Idavidrein/gpqa":           "gpqa_diamond",
        "EleutherAI/hendrycks_math": "all",
    }

    # Known split names to try (covers standard + non-standard)
    SPLIT_CANDIDATES = [
        "train", "test", "validation",
        "train_sft", "test_sft", "train_gen", "test_gen",  # ultrachat
        "dev",
    ]

    cfg = CONFIG_MAP.get(name)
    rows = []

    for split in SPLIT_CANDIDATES:
        try:
            if cfg:
                ds = load_dataset(name, cfg, split=split)
            else:
                ds = load_dataset(name, split=split)

            selected = ds if (limit_per_split is None or limit_per_split <= 0) \
                         else ds.select(range(min(limit_per_split, len(ds))))

            split_rows = []
            for item in selected:
                p = normalize_prompt(item, field_candidates)
                if len(p.strip()) > 0:
                    split_rows.append({
                        "source_dataset": name,
                        "split": split,
                        "prompt": p,
                    })

            if split_rows:
                print(f"  [{name}] split={split:12s}  {len(split_rows)} rows")
                rows.extend(split_rows)

        except Exception as e:
            # Split doesn't exist for this dataset — skip silently
            pass

    if not rows:
        raise RuntimeError(f"No splits loaded for {name}")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="Total rows to write (0 = all). Applied after loading all splits.")
    ap.add_argument("--limit_per_split", type=int, default=0,
                    help="Max rows per split per dataset (0 = all).")
    ap.add_argument("--config", default="configs/workloads_baselines.json")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    w = cfg[args.workload]

    lps = args.limit_per_split if args.limit_per_split > 0 else None

    all_rows = []
    for ds_name in w["dataset_hf"]:
        try:
            rows = load_hf_dataset_all_splits(ds_name, lps, w["prompt_field_candidates"])
            all_rows.extend(rows)
            print(f"  -> {len(rows)} total from {ds_name}")
        except Exception as e:
            print(f"[WARN] Failed loading {ds_name}: {e}")

    if not all_rows:
        print("[WARN] No data loaded — writing synthetic fallback prompts")
        all_rows = [
            {"source_dataset": "synthetic", "split": "synthetic",
             "prompt": f"Generate a detailed answer for {args.workload} example {i}."}
            for i in range(max(args.limit, 10))
        ]

    random.shuffle(all_rows)

    rows_to_write = all_rows if args.limit <= 0 else all_rows[:args.limit]

    out = Path(args.out_dir) / f"{args.workload}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, row in enumerate(rows_to_write):
            row["id"] = f"{args.workload}_{i:05d}"
            row["workload"] = args.workload
            f.write(json.dumps(row) + "\n")

    # Summary
    from collections import Counter
    split_counts = Counter(r["split"] for r in rows_to_write)
    ds_counts = Counter(r["source_dataset"] for r in rows_to_write)
    print(f"\nWrote {out} with {len(rows_to_write)} prompts")
    print(f"  By split:   {dict(split_counts)}")
    print(f"  By dataset: {dict(ds_counts)}")


if __name__ == "__main__":
    main()