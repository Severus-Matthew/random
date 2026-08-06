#!/usr/bin/env python
"""
split_by_key.py

Takes one or more .jsonl files and splits each into separate files
grouped by the value of the "split" key.

Usage:
    python split_by_key.py data/code_gen.jsonl data/mathematical_reasoning.jsonl
    python split_by_key.py data/*.jsonl
    python split_by_key.py data/code_gen.jsonl --out_dir data/splits
    python split_by_key.py data/code_gen.jsonl --key split   # default key
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def split_jsonl(input_path: Path, out_dir: Path, key: str):
    groups = defaultdict(list)
    skipped = 0

    with input_path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {input_path.name}:{lineno} — JSON parse error: {e}")
                skipped += 1
                continue

            val = obj.get(key)
            if val is None:
                groups["__no_split__"].append(obj)
            else:
                groups[str(val)].append(obj)

    if skipped:
        print(f"  [WARN] Skipped {skipped} malformed lines in {input_path.name}")

    stem = input_path.stem  # e.g. "code_gen"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for split_val, rows in sorted(groups.items()):
        out_path = out_dir / f"{stem}_{split_val}.jsonl"
        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        written[split_val] = (len(rows), out_path)
        print(f"  -> {out_path.name:50s}  {len(rows):>6} rows")

    return written


def main():
    ap = argparse.ArgumentParser(
        description="Split .jsonl files into per-split files based on a key value."
    )
    ap.add_argument("files", nargs="+", help="Input .jsonl file(s)")
    ap.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: same directory as each input file)"
    )
    ap.add_argument(
        "--key", default="split",
        help="JSON key to group by (default: 'split')"
    )
    args = ap.parse_args()

    for file_str in args.files:
        input_path = Path(file_str)
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}")
            continue
        if input_path.suffix != ".jsonl":
            print(f"[WARN] {input_path.name} doesn't look like a .jsonl file — processing anyway")

        out_dir = Path(args.out_dir) if args.out_dir else input_path.parent

        print(f"\n{input_path.name}")
        written = split_jsonl(input_path, out_dir, args.key)
        total = sum(n for n, _ in written.values())
        print(f"  Total: {total} rows -> {len(written)} file(s)")


if __name__ == "__main__":
    main()