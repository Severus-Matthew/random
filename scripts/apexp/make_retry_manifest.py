#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


GOOD = {"SUCCESS", "SKIPPED"}


def read_tsv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "NA") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    status = Path(args.status)
    out = Path(args.out)

    jobs = read_tsv(manifest)

    latest = {}
    if status.exists():
        for r in read_tsv(status):
            latest[r["job_id"]] = r["status"]

    retry = []
    for j in jobs:
        st = latest.get(j["job_id"], "MISSING_STATUS")
        if st not in GOOD:
            jj = dict(j)
            jj["previous_status"] = st
            retry.append(jj)

    fieldnames = list(jobs[0].keys())
    # Do not add previous_status to the manifest because the runner expects exact columns.
    retry_manifest_rows = [{k: r.get(k, "NA") for k in fieldnames} for r in retry]
    write_tsv(out, retry_manifest_rows, fieldnames)

    print("Original jobs:", len(jobs))
    print("Retry jobs:", len(retry_manifest_rows))
    print("Retry manifest:", out)

    if retry:
        print("\nFailed/missing jobs:")
        for r in retry[:50]:
            print(r["job_id"], "status=", r["previous_status"])
        if len(retry) > 50:
            print("... plus", len(retry) - 50, "more")


if __name__ == "__main__":
    main()
