#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np


def iter_jsonl(path: Path):
    for line in path.read_text(errors="ignore").splitlines():
        for part in line.split("\\n"):
            part = part.strip()
            if not part:
                continue
            try:
                yield json.loads(part)
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    counts = Counter()
    times = []
    active = Counter()
    nextk = Counter()

    for p in root.rglob("online_fast_trace.jsonl"):
        for e in iter_jsonl(p):
            typ = e.get("type")
            if typ == "learned_policy_init":
                counts[("init", e.get("loaded"), e.get("error", ""))] += 1
            elif typ == "choose_active_k_for_scheduler":
                active[(e.get("mode"), e.get("active_k"))] += 1
            elif typ == "observe_block_after_verify_update_future_k":
                counts[("observe", e.get("mode"), e.get("learned_policy_used"), e.get("policy_error", ""))] += 1
                if e.get("learned_policy_used"):
                    t = e.get("policy_runtime_ms")
                    if t is not None:
                        times.append(float(t))
                    nextk[e.get("next_k_for_future_unscheduled_block")] += 1

    print("COUNTS")
    for k, v in counts.most_common(100):
        print(k, v)

    print("\nACTIVE K")
    for k, v in active.most_common(100):
        print(k, v)

    print("\nNEXT K FROM LEARNED")
    for k, v in nextk.most_common(100):
        print(k, v)

    print("\nPOLICY RUNTIME MS")
    print("n:", len(times))
    if times:
        arr = np.array(times)
        print("mean:", float(arr.mean()))
        print("p50:", float(np.percentile(arr, 50)))
        print("p95:", float(np.percentile(arr, 95)))
        print("max:", float(arr.max()))


if __name__ == "__main__":
    main()
