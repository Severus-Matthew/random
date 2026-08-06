#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


PROMPT_TEXT_COLS = [
    "prompt",
    "prompt_text",
    "input",
    "input_text",
    "question",
    "problem",
    "instruction",
    "text",
]

ID_COLS = [
    "id",
    "prompt_id",
    "task_id",
    "problem_id",
    "sample_id",
    "question_id",
    "idx",
    "index",
]


KNOWN_WORKLOADS = [
    "code_gen",
    "hardware_gen",
    "long_chain_reasoning",
    "long_context_completion",
    "long_horizon_swe",
    "mathematical_reasoning",
    "conversational_generation_gen",
    "conversational_generation_sft",
    "conversational_generation",
]


def sha256_hex(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8", errors="ignore")).hexdigest()


def hash_variants(text: str):
    raw = text or ""
    stripped = raw.strip()
    collapsed = " ".join(stripped.split())

    variants = set()
    for s in [raw, stripped, collapsed]:
        h = sha256_hex(s)
        variants.update([h, h[:64], h[:40], h[:32], h[:16], h[:12]])
    return variants


def stringify_prompt(v):
    if v is None:
        return None

    if isinstance(v, str):
        return v if v.strip() else None

    if isinstance(v, list):
        parts = []
        for m in v:
            if isinstance(m, dict):
                role = str(m.get("role", ""))
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(x) for x in content)
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(m))
        txt = "\n".join(parts).strip()
        return txt if txt else None

    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)

    return str(v)


def extract_prompt_text(record):
    if not isinstance(record, dict):
        return stringify_prompt(record)

    for c in PROMPT_TEXT_COLS:
        if c in record:
            txt = stringify_prompt(record.get(c))
            if txt:
                return txt

    for c in ["messages", "conversation", "conversations", "turns"]:
        if c in record:
            txt = stringify_prompt(record.get(c))
            if txt:
                return txt

    # Fallback: concatenate likely input fields, but avoid output/completion fields.
    parts = []
    for k, v in record.items():
        lk = str(k).lower()
        if any(bad in lk for bad in ["output", "completion", "answer", "response", "generated"]):
            continue
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v}")

    txt = "\n".join(parts).strip()
    return txt if txt else None


def infer_workload_aliases(path: Path):
    s = str(path)
    aliases = set()

    for w in KNOWN_WORKLOADS:
        if w in s:
            aliases.add(w)

    # Your prompt IDs use conversational_generation_XXXXX even when workload is
    # conversational_generation_gen or conversational_generation_sft.
    if "conversational_generation_gen" in s or "conversational_generation_sft" in s:
        aliases.add("conversational_generation")

    return sorted(aliases)


def add_candidate_id(ids: set[str], x, workload_aliases):
    if x is None:
        return

    xs = str(x).strip()
    if not xs:
        return

    ids.add(xs)

    # If id is numeric, generate workload_00004-style aliases.
    if re.fullmatch(r"\d+", xs):
        n = int(xs)
        for w in workload_aliases:
            ids.add(f"{w}_{n}")
            ids.add(f"{w}_{n:05d}")
            ids.add(f"{w}_{n:06d}")


def candidate_ids_for_record(record, row_idx: int, workload_aliases):
    ids = set()

    if isinstance(record, dict):
        for c in ID_COLS:
            if c in record:
                add_candidate_id(ids, record.get(c), workload_aliases)

    # Also generate IDs from row order. Include both 0-based and 1-based just in case.
    for w in workload_aliases:
        ids.add(f"{w}_{row_idx}")
        ids.add(f"{w}_{row_idx:05d}")
        ids.add(f"{w}_{row_idx:06d}")

        ids.add(f"{w}_{row_idx + 1}")
        ids.add(f"{w}_{row_idx + 1:05d}")
        ids.add(f"{w}_{row_idx + 1:06d}")

    return ids


def scan_benchmark_jsonl(benchmark_dir: Path):
    rows = []

    files = sorted(benchmark_dir.rglob("*.jsonl"))
    print(f"Scanning {len(files)} jsonl files under {benchmark_dir}")

    for p in files:
        aliases = infer_workload_aliases(p)

        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except Exception:
                    continue

                prompt_text = extract_prompt_text(record)
                if not prompt_text:
                    continue

                candidate_ids = candidate_ids_for_record(record, i, aliases)

                # Hash candidates are fallback only.
                for hv in hash_variants(prompt_text):
                    candidate_ids.add(hv)

                for cid in candidate_ids:
                    rows.append({
                        "candidate_key": str(cid),
                        "prompt_text": prompt_text,
                        "source_file": str(p),
                        "row_idx": i,
                        "workload_aliases": ",".join(aliases),
                    })

    if not rows:
        raise RuntimeError(f"No prompt candidates extracted from {benchmark_dir}")

    cand = pd.DataFrame(rows)
    cand = cand.drop_duplicates("candidate_key", keep="first")
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--benchmark-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-out", required=True)
    ap.add_argument("--min-match-rate", type=float, default=0.90)
    args = ap.parse_args()

    data_path = Path(args.data)
    benchmark_dir = Path(args.benchmark_dir)
    out_path = Path(args.out)
    audit_path = Path(args.audit_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    online = pd.read_csv(
        data_path,
        usecols=lambda c: c in {"prompt_hash", "prompt_id", "workload", "source_dataset", "split"},
        low_memory=False,
    )

    if "prompt_hash" not in online.columns:
        raise RuntimeError("online dataset must contain prompt_hash")
    if "prompt_id" not in online.columns:
        raise RuntimeError("online dataset must contain prompt_id")

    online["prompt_hash"] = online["prompt_hash"].astype(str)
    online["prompt_id"] = online["prompt_id"].astype(str)

    needed = (
        online[["prompt_hash", "prompt_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("Unique prompt_hash needed:", needed["prompt_hash"].nunique())
    print("Unique prompt_id needed:", needed["prompt_id"].nunique())

    cand = scan_benchmark_jsonl(benchmark_dir)

    # First try prompt_id match.
    by_pid = needed.merge(
        cand,
        left_on="prompt_id",
        right_on="candidate_key",
        how="left",
    )
    by_pid["match_method"] = "prompt_id"

    # For unmatched rows, try prompt_hash fallback.
    unmatched = by_pid[by_pid["prompt_text"].isna()][["prompt_hash", "prompt_id"]].copy()

    if len(unmatched):
        by_hash = unmatched.merge(
            cand,
            left_on="prompt_hash",
            right_on="candidate_key",
            how="left",
        )
        by_hash["match_method"] = "prompt_hash_or_hash_variant"

        matched_pid = by_pid[by_pid["prompt_text"].notna()].copy()
        matched = pd.concat([matched_pid, by_hash], ignore_index=True)
    else:
        matched = by_pid

    matched["matched"] = matched["prompt_text"].notna()

    rate_hash = (
        matched[["prompt_hash", "matched"]]
        .drop_duplicates("prompt_hash")["matched"]
        .mean()
    )
    rate_id = (
        matched[["prompt_id", "matched"]]
        .drop_duplicates("prompt_id")["matched"]
        .mean()
    )

    print(f"Matched by prompt_hash coverage: {rate_hash:.4f}")
    print(f"Matched by prompt_id coverage:   {rate_id:.4f}")
    print(f"Matched rows: {matched['matched'].sum()} / {len(matched)}")

    matched.to_csv(audit_path, index=False)

    if rate_hash < args.min_match_rate:
        print("\nUnmatched examples:")
        cols = ["prompt_hash", "prompt_id", "matched"]
        print(matched[~matched["matched"]][cols].head(30).to_string(index=False))
        raise RuntimeError(
            f"Prompt text match rate {rate_hash:.4f} below required {args.min_match_rate}. "
            f"Inspect {audit_path}."
        )

    out = (
        matched[matched["matched"]][["prompt_hash", "prompt_text"]]
        .drop_duplicates("prompt_hash")
        .reset_index(drop=True)
    )

    out.to_csv(out_path, index=False)

    print("Wrote prompt text map:", out_path)
    print("Wrote audit:", audit_path)
    print("Output shape:", out.shape)

    print("\nSample recovered prompts:")
    print(out.head(3).to_string(index=False, max_colwidth=160))


if __name__ == "__main__":
    main()
