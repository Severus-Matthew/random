#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOTS = [
    Path("slurm_logs/evaluation_jobs"),
]

OUT = Path("slurm_logs/evaluation_jobs/failure_audit")
OUT.mkdir(parents=True, exist_ok=True)

TIME_PAT = re.compile(
    r"(FAILED DUE TO TIME LIMIT|DUE TO TIME LIMIT|TIME LIMIT|TIMEOUT|timed out|CANCELLED AT.*DUE TO TIME LIMIT)",
    re.IGNORECASE,
)

MEM_PAT = re.compile(
    r"(CUDA out of memory|out of memory|OutOfMemoryError|OUT_OF_MEMORY|oom|OOM|Killed|exceeded memory|Cannot allocate memory|CUBLAS_STATUS_ALLOC_FAILED|cuda.*memory|vllm.*memory)",
    re.IGNORECASE,
)

ERR_PAT = re.compile(
    r"(Traceback \(most recent call last\)|RuntimeError:|ValueError:|AttributeError:|ImportError:|ModuleNotFoundError:|AssertionError:|worker_failed|FAILED|Error:)",
    re.IGNORECASE,
)


def read_tail(path: Path, nbytes: int = 16000) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    return data[-nbytes:].decode("utf-8", errors="ignore")


def read_all_small_or_tail(path: Path, nbytes: int = 256000) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if len(data) <= nbytes:
        return data.decode("utf-8", errors="ignore")
    return data[-nbytes:].decode("utf-8", errors="ignore")


def extract_job_id(path: Path) -> str:
    # Common Slurm patterns end with _12345.out, _12345.err, slurm-12345.out
    s = path.name
    m = re.search(r"(?:_|-|\.)(\d{3,})(?:\.(?:out|err)|$)", s)
    if m:
        return m.group(1)
    return ""


def extract_job_name(path: Path) -> str:
    # Remove job id and extension from log filename.
    name = path.name
    name = re.sub(r"\.(out|err)$", "", name)
    name = re.sub(r"[_-]\d{3,}$", "", name)
    name = re.sub(r"^slurm[-_]", "", name)
    return name


def first_match(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    line_start = text.rfind("\n", 0, m.start()) + 1
    line_end = text.find("\n", m.end())
    if line_end < 0:
        line_end = len(text)
    return text[line_start:line_end].strip()[:500]


def gather_logs():
    files = []
    seen = set()
    for root in ROOTS:
        if not root.exists():
            continue
        for ext in ("*.err", "*.out"):
            for p in root.rglob(ext):
                if p in seen:
                    continue
                seen.add(p)
                files.append(p)
    return sorted(files)


def find_sbatch_for_jobname(job_name: str) -> str:
    if not job_name:
        return ""
    for p in sorted(Path("slurm_logs/evaluation_jobs").glob("*.sbatch")):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for line in txt.splitlines():
            if line.strip().startswith("#SBATCH") and "--job-name=" in line:
                jn = line.split("--job-name=", 1)[1].strip().strip('"').strip("'")
                if jn == job_name:
                    return str(p)
    return ""


def write_tsv(path: Path, rows: list[dict]):
    cols = ["job_id", "job_name", "kind", "log_file", "sbatch_file", "evidence"]
    with path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")).replace("\t", " ") for c in cols) + "\n")


def write_rerun_script(path: Path, rows: list[dict]):
    scripts = []
    seen = set()
    for r in rows:
        s = r.get("sbatch_file", "")
        if s and s not in seen:
            seen.add(s)
            scripts.append(s)

    with path.open("w") as f:
        f.write("#!/bin/bash\nset -euo pipefail\n\n")
        for s in scripts:
            f.write(f"sbatch {s}\n")

    path.chmod(0o755)
    return scripts


def main():
    time_rows = []
    mem_rows = []
    other_rows = []

    for p in gather_logs():
        text = read_all_small_or_tail(p)
        tail = read_tail(p)

        job_id = extract_job_id(p)
        job_name = extract_job_name(p)
        sbatch = find_sbatch_for_jobname(job_name)

        base = {
            "job_id": job_id,
            "job_name": job_name,
            "log_file": str(p),
            "sbatch_file": sbatch,
        }

        tm = first_match(TIME_PAT, text)
        mm = first_match(MEM_PAT, text)
        ee = first_match(ERR_PAT, tail)

        if tm:
            time_rows.append({**base, "kind": "TIME_LIMIT", "evidence": tm})
        elif mm:
            mem_rows.append({**base, "kind": "MEMORY_OOM", "evidence": mm})
        elif ee and p.suffix == ".out":
            # Only treat .out tail errors as "other" if it looks like the job ended in an exception.
            other_rows.append({**base, "kind": "OTHER_ERROR", "evidence": ee})

    # Deduplicate by job_id/job_name/kind/log.
    def dedup(rows):
        seen = set()
        out = []
        for r in rows:
            key = (r["job_id"], r["job_name"], r["kind"], r["log_file"])
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    time_rows = dedup(time_rows)
    mem_rows = dedup(mem_rows)
    other_rows = dedup(other_rows)

    write_tsv(OUT / "time_limit_jobs.tsv", time_rows)
    write_tsv(OUT / "memory_error_jobs.tsv", mem_rows)
    write_tsv(OUT / "other_error_jobs.tsv", other_rows)
    write_tsv(OUT / "all_failed_classified_jobs.tsv", time_rows + mem_rows + other_rows)

    time_scripts = write_rerun_script(OUT / "rerun_time_limit_only.sh", time_rows)
    mem_scripts = write_rerun_script(OUT / "rerun_memory_error_only.sh", mem_rows)
    all_scripts = write_rerun_script(OUT / "rerun_all_classified_failed.sh", time_rows + mem_rows + other_rows)

    print("WROTE:", OUT)
    print("time-limit log hits:", len(time_rows))
    print("memory/OOM log hits:", len(mem_rows))
    print("other-error log hits:", len(other_rows))
    print("time-limit sbatch scripts:", len(time_scripts))
    print("memory/OOM sbatch scripts:", len(mem_scripts))
    print("all classified sbatch scripts:", len(all_scripts))
    print()
    print("TIME LIMIT LIST:")
    for r in time_rows:
        print(f"{r['job_id']}\t{r['job_name']}\t{r['log_file']}\t{r['sbatch_file']}")
    print()
    print("MEMORY/OOM LIST:")
    for r in mem_rows:
        print(f"{r['job_id']}\t{r['job_name']}\t{r['log_file']}\t{r['sbatch_file']}")


if __name__ == "__main__":
    main()
