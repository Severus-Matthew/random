#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except Exception as e:
                yield i, {"__parse_error__": repr(e), "__raw__": line}


def validate_event(e):
    errors = []

    if "__parse_error__" in e:
        return ["json_parse_error"]

    required = ["req_id", "k_actual", "accepted_len", "first_rejection", "accepted_mask"]
    for r in required:
        if r not in e:
            errors.append(f"missing_{r}")

    if errors:
        return errors

    k = e["k_actual"]
    L = e["accepted_len"]
    R = e["first_rejection"]
    mask = e["accepted_mask"]

    if not isinstance(k, int) or k < 0:
        errors.append("bad_k_actual")
    if not isinstance(L, int) or L < 0:
        errors.append("bad_accepted_len")
    if not isinstance(R, int) or R < 0:
        errors.append("bad_first_rejection")
    if isinstance(k, int) and isinstance(L, int) and L > k:
        errors.append("accepted_len_gt_k")
    if isinstance(k, int) and isinstance(R, int) and R > k:
        errors.append("first_rejection_gt_k")

    if isinstance(mask, list) and isinstance(k, int):
        if len(mask) != k:
            errors.append("mask_len_ne_k")
        if isinstance(L, int) and 0 <= L <= k:
            expected = [True] * L + [False] * (k - L)
            if mask != expected:
                errors.append("mask_not_prefix")
    else:
        errors.append("bad_mask")

    if isinstance(k, int) and isinstance(L, int) and isinstance(R, int):
        if L == k and R != k:
            errors.append("full_accept_R_ne_k")
        if L < k and R != L:
            errors.append("partial_accept_R_ne_L")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    path = Path(args.path)
    total = 0
    err_counts = Counter()
    by_method = Counter()
    by_k = Counter()
    accepted_hist = Counter()
    examples = defaultdict(list)

    for line_no, e in load_jsonl(path):
        total += 1
        errs = validate_event(e)
        if errs:
            for er in errs:
                err_counts[er] += 1
                if len(examples[er]) < 3:
                    examples[er].append({"line": line_no, "event": e})
        else:
            by_method[e.get("method", "unknown")] += 1
            by_k[e.get("k_actual", "unknown")] += 1
            accepted_hist[e.get("accepted_len", "unknown")] += 1

    print("file:", path)
    print("total_events:", total)
    print("valid_events:", total - sum(err_counts.values()) if not err_counts else "see error counts")
    print()
    print("error_counts:")
    for k, v in err_counts.most_common():
        print(f"  {k}: {v}")
    print()
    print("by_method:")
    for k, v in by_method.most_common():
        print(f"  {k}: {v}")
    print()
    print("by_k:")
    for k, v in sorted(by_k.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")
    print()
    print("accepted_len_hist:")
    for k, v in sorted(accepted_hist.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")

    if err_counts:
        print()
        print("error_examples:")
        print(json.dumps(examples, indent=2)[:5000])


if __name__ == "__main__":
    main()
