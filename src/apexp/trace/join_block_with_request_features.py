#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd


PROMPT_FEATURE_COLS = [
    "id",
    "prompt_id",
    "request_ordinal",
    "prompt_hash",
    "prompt_char_len",
    "prompt_token_len",
    "vllm_request_id",
    "workload",
    "run_name",
    "experiment",
    "source_dataset",
    "split",
    "method",
    "k",
    "temperature",
    "seed",
    "draft_model",
    "eagle3_model",
    "ngram_lookup_min",
    "ngram_lookup_max",
    "max_prompt_tokens",
    "num_turns",
    "num_turns_completed",
    "n_output_tokens",
    "latency_s",
    "tokens_per_sec",
    "mean_entropy",
    "repetition_density",
    "entropy_bucket",
    "accepted_per_draft",
    "acceptance_rate",
    "draft_tokens",
    "accepted_tokens_total",
]


def _is_missing_series(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.lower().isin(["", "nan", "none", "null"]))


def _normalize_req_id(x):
    if pd.isna(x):
        return None
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _req_prefix(x):
    s = _normalize_req_id(x)
    if s is None:
        return None
    return s.split("-", 1)[0]


def coalesce_columns(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("prompt_id", "prompt_id_summary"),
        ("prompt_hash", "prompt_hash_summary"),
        ("workload", "workload_summary"),
        ("method", "method_summary"),
        ("temperature", "temperature_summary"),
        ("seed", "seed_summary"),
    ]

    for base, summ in pairs:
        if summ in df.columns:
            if base not in df.columns:
                df[base] = df[summ]
            else:
                mask = _is_missing_series(df[base])
                df.loc[mask, base] = df.loc[mask, summ]

    if "prompt_id" in df.columns and "id" in df.columns:
        mask = _is_missing_series(df["prompt_id"])
        df.loc[mask, "prompt_id"] = df.loc[mask, "id"]

    if "prompt_hash" in df.columns and "prompt_hash_summary" in df.columns:
        mask = _is_missing_series(df["prompt_hash"])
        df.loc[mask, "prompt_hash"] = df.loc[mask, "prompt_hash_summary"]

    return df


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def first_seen_req_order(block_df: pd.DataFrame):
    od = OrderedDict()
    for r in block_df["req_id"].astype(str):
        if r not in od:
            od[r] = len(od)
    return od


def load_summaries(trace_dir: Path) -> pd.DataFrame:
    files = sorted(trace_dir.rglob("summary.csv"))
    if not files:
        raise FileNotFoundError(f"No summary.csv found under {trace_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["summary_file"] = str(f)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if "request_ordinal" not in out.columns:
        out["request_ordinal"] = range(len(out))

    if "prompt_id" not in out.columns and "id" in out.columns:
        out["prompt_id"] = out["id"].astype(str)

    return out


def infer_turns(summary: pd.DataFrame, n_unique_req: int) -> int:
    if "num_turns" in summary.columns:
        vals = pd.to_numeric(summary["num_turns"], errors="coerce").dropna().unique()
        vals = [int(v) for v in vals if int(v) >= 1]
        if len(vals) == 1:
            return vals[0]

    if "num_turns_completed" in summary.columns:
        vals = pd.to_numeric(summary["num_turns_completed"], errors="coerce").dropna().unique()
        vals = [int(v) for v in vals if int(v) >= 1]
        if len(vals) == 1:
            return vals[0]

    n_summary = max(1, len(summary))
    ratio = round(n_unique_req / n_summary)
    return max(1, int(ratio))


def prompt_match_rate(df: pd.DataFrame) -> float:
    if "prompt_id" not in df.columns:
        return 0.0
    return float(df["prompt_id"].notna().mean())


def join_one(trace_dir: Path):
    block_path = trace_dir / "block_events.jsonl"
    if not block_path.exists():
        raise FileNotFoundError(f"Missing {block_path}")

    block = read_jsonl(block_path)
    if len(block) == 0:
        raise RuntimeError(f"No block events in {block_path}")

    block["req_id"] = block["req_id"].astype(str)
    block["trace_dir"] = str(trace_dir)
    block["_req_id_norm"] = block["req_id"].map(_normalize_req_id)
    block["_req_id_prefix"] = block["req_id"].map(_req_prefix)

    summary = load_summaries(trace_dir)
    keep = [c for c in PROMPT_FEATURE_COLS + ["summary_file"] if c in summary.columns]
    summary_small = summary[keep].copy()

    if "vllm_request_id" in summary_small.columns:
        summary_small["_vllm_request_id_norm"] = summary_small["vllm_request_id"].map(_normalize_req_id)
        summary_small["_vllm_request_id_prefix"] = summary_small["vllm_request_id"].map(_req_prefix)

    joined = None
    join_mode = None
    exact_match_rate = 0.0

    # 1. Exact vLLM request id.
    if "vllm_request_id" in summary_small.columns:
        exact = block.merge(
            summary_small,
            left_on="_req_id_norm",
            right_on="_vllm_request_id_norm",
            how="left",
            suffixes=("", "_summary"),
        )
        exact = coalesce_columns(exact)
        exact_match_rate = prompt_match_rate(exact)

        if exact_match_rate >= 0.95:
            joined = exact
            join_mode = "exact_vllm_request_id"

    # 2. Prefix match. Block req_id can look like "0-af5d7bf3" while summary has "0".
    if joined is None and "vllm_request_id" in summary_small.columns:
        prefix = block.merge(
            summary_small,
            left_on="_req_id_prefix",
            right_on="_vllm_request_id_norm",
            how="left",
            suffixes=("", "_summary"),
        )
        prefix = coalesce_columns(prefix)
        prefix_match_rate = prompt_match_rate(prefix)

        if prefix_match_rate >= 0.95:
            joined = prefix
            join_mode = "exact_vllm_request_id_prefix"
            exact_match_rate = prefix_match_rate

    # 3. Order fallback, including multi-turn.
    if joined is None:
        req_order = first_seen_req_order(block)
        block["_request_order"] = block["req_id"].map(req_order)

        n_unique_req = len(req_order)
        turns = infer_turns(summary_small, n_unique_req)

        # In the normal single-turn case:
        #   summary row i corresponds to vLLM request i.
        # In multi-turn:
        #   one summary row aggregates turns, while block events have one vLLM request per turn.
        #   summary row i corresponds to request orders [i*turns, ..., i*turns + turns - 1].
        block["_request_ordinal_fallback"] = (block["_request_order"] // max(1, turns)).astype(int)

        summary_small = summary_small.sort_values("request_ordinal").copy()
        summary_small["_request_ordinal_fallback"] = range(len(summary_small))

        joined = block.merge(
            summary_small,
            on="_request_ordinal_fallback",
            how="left",
            suffixes=("", "_summary"),
        )
        joined = coalesce_columns(joined)
        join_mode = f"order_fallback_turns_{turns}"

    pmr = prompt_match_rate(joined)

    diagnostics = {
        "trace_dir": str(trace_dir),
        "n_block_rows": int(len(block)),
        "n_unique_block_req_ids": int(block["req_id"].nunique()),
        "n_summary_rows": int(len(summary)),
        "summary_files": sorted(summary["summary_file"].unique().tolist()),
        "join_mode": join_mode,
        "exact_match_rate": float(exact_match_rate),
        "prompt_match_rate": float(pmr),
        "unmatched_rows": int(joined["prompt_id"].isna().sum()) if "prompt_id" in joined.columns else int(len(joined)),
    }

    return joined, diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_joined = []
    all_diag = []

    for d in args.trace_dirs:
        joined, diag = join_one(Path(d))
        all_joined.append(joined)
        all_diag.append(diag)

    full = pd.concat(all_joined, ignore_index=True)
    full.to_csv(out_dir / "block_with_request_features.csv", index=False)

    with (out_dir / "join_diagnostics.json").open("w") as f:
        json.dump(all_diag, f, indent=2)

    print(json.dumps(all_diag, indent=2))
    print("Wrote:", out_dir / "block_with_request_features.csv")


if __name__ == "__main__":
    main()
