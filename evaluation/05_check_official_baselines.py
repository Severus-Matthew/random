#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from evaluation.apex_eval_utils import ensure_dir, write_json


OFFICIAL_BASELINES = {
    "BanditSpec": {
        "status": "recommended_official_if_repo_available",
        "env": "BANDITSPEC_ROOT",
        "why": "Closest adaptive online speculative-decoding hyperparameter baseline. Use official repo when present; otherwise report as related work and do not silently substitute an unofficial implementation.",
    },
    "DSDE": {
        "status": "related_work_no_official_repo_configured",
        "env": None,
        "why": "Closely related dynamic speculation-length method, but no official implementation is configured in this harness.",
    },
    "AdaEAGLE": {
        "status": "related_work_unless_official_repo_supplied",
        "env": "ADAEAGLE_ROOT",
        "why": "Learned adaptive EAGLE depth/structure baseline. Use only if an official runnable repo is supplied.",
    },
    "HeteroSpec": {
        "status": "related_work_unless_official_repo_supplied",
        "env": "HETEROSPEC_ROOT",
        "why": "Entropy/heterogeneity-based adaptive speculation. It is conceptually relevant but not the same vLLM k-controller unless official code is available.",
    },
}


def git_head(path: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return None


def inspect_repo(root: Path) -> dict:
    files = []
    for p in root.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.is_file():
            files.append(str(p.relative_to(root)))
        if len(files) >= 500:
            break
    return {
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "git_head": git_head(root),
        "sample_files": files[:80],
        "has_readme": any(Path(f).name.lower().startswith("readme") for f in files),
        "has_python": any(f.endswith(".py") for f in files),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = ensure_dir(args.out_dir)

    report = {}
    for name, spec in OFFICIAL_BASELINES.items():
        env = spec.get("env")
        entry = dict(spec)
        if env:
            val = os.environ.get(env, "")
            entry["env_value"] = val
            if val:
                entry["repo_inspection"] = inspect_repo(Path(val))
                ok = entry["repo_inspection"].get("exists") and entry["repo_inspection"].get("has_python")
                entry["runnable_status"] = "present_needs_manual_api_mapping" if ok else "path_invalid_or_no_python_files"
            else:
                entry["runnable_status"] = "not_configured"
        else:
            entry["runnable_status"] = "not_configured_related_work_only"
        report[name] = entry

    write_json(out_dir / "official_baseline_readiness.json", report)

    md = ["# Official external baseline readiness", ""]
    for name, entry in report.items():
        md.append(f"## {name}")
        md.append(f"- status: `{entry['status']}`")
        md.append(f"- runnable_status: `{entry['runnable_status']}`")
        if entry.get("env"):
            md.append(f"- env: `{entry['env']}`")
            md.append(f"- env_value: `{entry.get('env_value', '')}`")
        md.append(f"- why: {entry['why']}")
        insp = entry.get("repo_inspection")
        if insp:
            md.append(f"- git_head: `{insp.get('git_head')}`")
            md.append(f"- has_python: `{insp.get('has_python')}`")
            md.append(f"- has_readme: `{insp.get('has_readme')}`")
        md.append("")
    (out_dir / "official_baseline_readiness.md").write_text("\n".join(md))

    print("WROTE", out_dir / "official_baseline_readiness.json")
    print("WROTE", out_dir / "official_baseline_readiness.md")
    print("\nImportant: this script intentionally does not invent unofficial implementations for missing official baselines.")


if __name__ == "__main__":
    main()
