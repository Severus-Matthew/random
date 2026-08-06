#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def repetition_density(tokens):
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / max(1, len(tokens)))


def mean_or_zero(xs):
    xs = [safe_float(x) for x in xs]
    xs = [x for x in xs if math.isfinite(x)]
    return float(np.mean(xs)) if xs else 0.0


def std_or_zero(xs):
    xs = [safe_float(x) for x in xs]
    xs = [x for x in xs if math.isfinite(x)]
    return float(np.std(xs)) if xs else 0.0



def request_keys_from_trace_row(r):
    keys = []
    for k in ["vllm_request_id", "prompt_hash", "prompt_id", "id"]:
        v = r.get(k)
        if v is None:
            continue
        sv = str(v)
        if sv in ["", "nan", "None", "null"]:
            continue
        keys.append(sv)
        if sv.endswith(".0"):
            keys.append(sv[:-2])
        keys.append(sv.split("-", 1)[0])

    out = []
    seen = set()
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def load_traces(trace_root: Path):
    """
    Map traces by both nested trace directory and top-level job root.

    block rows store:
      results/.../runs/<job_id>

    traces live at:
      results/.../runs/<job_id>/<experiment>/<run_name>/<method>/<config>/traces.jsonl
    """
    traces = defaultdict(dict)

    files = sorted(trace_root.rglob("traces.jsonl"))
    print("Found traces.jsonl files:", len(files))

    for tf in files:
        try:
            rel = tf.relative_to(trace_root)
            job_id = rel.parts[0]
            job_root = trace_root / job_id
        except Exception:
            job_id = None
            job_root = tf.parent

        path_keys = [
            str(tf.parent),
            str(tf.parent.resolve()),
            str(job_root),
            str(job_root.resolve()),
        ]

        if job_id:
            path_keys.append(job_id)

        # Deduplicate path keys.
        tmp = []
        seen = set()
        for k in path_keys:
            if k and k not in seen:
                tmp.append(k)
                seen.add(k)
        path_keys = tmp

        with tf.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)

                req_keys = request_keys_from_trace_row(r)

                for pk in path_keys:
                    for rk in req_keys:
                        traces[pk][rk] = r

    return dict(traces)


def find_trace_for_block(row, trace_map):
    raw_trace_dir = str(row.get("trace_dir", ""))

    trace_dir_candidates = [
        raw_trace_dir,
        Path(raw_trace_dir).name,
    ]

    try:
        trace_dir_candidates.append(str(Path(raw_trace_dir).resolve()))
    except Exception:
        pass

    # Deduplicate trace-dir candidates.
    tmp = []
    seen = set()
    for k in trace_dir_candidates:
        if k and k not in seen:
            tmp.append(k)
            seen.add(k)
    trace_dir_candidates = tmp

    req_key_candidates = []
    for k in ["vllm_request_id", "req_id", "prompt_hash", "prompt_id", "id"]:
        if k not in row:
            continue
        v = row.get(k)
        if pd.isna(v):
            continue
        sv = str(v)
        if sv in ["", "nan", "None", "null"]:
            continue

        req_key_candidates.append(sv)
        if sv.endswith(".0"):
            req_key_candidates.append(sv[:-2])
        req_key_candidates.append(sv.split("-", 1)[0])

    # Deduplicate request-key candidates.
    tmp = []
    seen = set()
    for k in req_key_candidates:
        if k and k not in seen:
            tmp.append(k)
            seen.add(k)
    req_key_candidates = tmp

    for td in trace_dir_candidates:
        reqs = trace_map.get(td)
        if not reqs:
            continue

        for rk in req_key_candidates:
            if rk in reqs:
                return reqs[rk]

    return None


def prefix_features(token_ids, entropy, pos):
    n = len(token_ids)
    pos = max(0, min(int(pos), n))

    out = {
        "prefix_pos": pos,
        "prefix_frac": pos / max(1, n),
        "prefix_entropy_mean_all": 0.0,
        "prefix_entropy_std_all": 0.0,
        "prefix_repetition_all": 0.0,
        "prefix_len_nonzero": int(pos > 0),
    }

    prefix_tokens = token_ids[:pos]
    prefix_entropy = entropy[:pos]

    out["prefix_entropy_mean_all"] = mean_or_zero(prefix_entropy)
    out["prefix_entropy_std_all"] = std_or_zero(prefix_entropy)
    out["prefix_repetition_all"] = repetition_density(prefix_tokens)

    for w in [16, 32, 64, 128, 256]:
        s = max(0, pos - w)
        toks = token_ids[s:pos]
        ents = entropy[s:pos]

        out[f"prefix_entropy_mean_w{w}"] = mean_or_zero(ents)
        out[f"prefix_entropy_std_w{w}"] = std_or_zero(ents)
        out[f"prefix_repetition_w{w}"] = repetition_density(toks)
        out[f"prefix_window_len_w{w}"] = len(toks)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--core-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "week2" / "online_block_dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    block_path = root / "block_joined" / "block_with_request_features.csv"
    if not block_path.exists():
        raise SystemExit(f"Missing {block_path}")

    print("Loading block dataset:", block_path)
    block = pd.read_csv(block_path, low_memory=False)

    print("Block rows:", block.shape)

    # Strictly remove AR from block training because AR has no speculative block events.
    block = block[block["method"].isin(["ngram_sd", "draft_sd", "eagle3"])].copy()

    if args.core_only:
        if "temperature" in block.columns:
            block = block[pd.to_numeric(block["temperature"], errors="coerce").fillna(0.0).eq(0.0)]
        if "max_prompt_tokens" in block.columns:
            block = block[pd.to_numeric(block["max_prompt_tokens"], errors="coerce").fillna(0).eq(0)]
        if "num_turns" in block.columns:
            block = block[pd.to_numeric(block["num_turns"], errors="coerce").fillna(1).eq(1)]

    # Keep columns numeric where needed.
    for c in [
        "accepted_len", "first_rejection", "num_rejected", "k_actual", "k",
        "temperature", "prompt_token_len", "latency_s", "tokens_per_sec",
        "n_output_tokens", "block_index",
    ]:
        if c in block.columns:
            block[c] = pd.to_numeric(block[c], errors="coerce")

    # Load per-token traces.
    print("Loading traces...")
    trace_map = load_traces(root / "runs")
    print("Trace dirs:", len(trace_map))

    # Stable ordering within each request/candidate trajectory.
    sort_cols = []
    for c in ["trace_dir", "req_id", "block_index"]:
        if c in block.columns:
            sort_cols.append(c)
    if "block_index" not in sort_cols:
        sort_cols += ["trace_dir", "req_id"]
    block = block.sort_values(sort_cols).reset_index(drop=True)

    group_cols = ["trace_dir", "req_id"]
    rows = []

    n_no_trace = 0
    n_total = 0

    for _, g in block.groupby(group_cols, dropna=False, sort=False):
        g = g.sort_values("block_index") if "block_index" in g.columns else g

        past_accepted = []
        past_full = []
        past_first_rej = []
        past_num_rej = []

        prefix_pos = 0

        for _, r in g.iterrows():
            n_total += 1

            tr = find_trace_for_block(r, trace_map)

            token_ids = []
            entropy = []

            if tr is not None:
                token_ids = tr.get("token_ids") or []
                entropy = tr.get("entropy") or []
            else:
                n_no_trace += 1

            # Causal prefix features use only tokens before this block.
            pf = prefix_features(token_ids, entropy, prefix_pos)

            accepted_len = safe_int(r.get("accepted_len"), 0)
            k_actual = safe_int(r.get("k_actual"), safe_int(r.get("k"), 1))
            first_rej = safe_int(r.get("first_rejection"), accepted_len)
            num_rej = safe_int(r.get("num_rejected"), max(0, k_actual - accepted_len))
            full_accept = int(accepted_len >= k_actual)

            outrow = {}

            # Identifiers / metadata.
            for c in [
                "workload", "method", "candidate_id", "trace_dir", "req_id",
                "prompt_id", "prompt_hash", "vllm_request_id",
                "experiment", "run_name", "source_dataset", "split",
                "draft_model", "eagle3_model",
                "ngram_lookup_min", "ngram_lookup_max",
            ]:
                if c in r.index:
                    outrow[c] = r.get(c)

            outrow.update({
                "block_index": safe_int(r.get("block_index"), 0),
                "target_accepted_len": accepted_len,
                "target_first_rejection": first_rej,
                "target_full_accept": full_accept,
                "target_num_rejected": num_rej,
                "target_k_actual": k_actual,
                "target_latency_s": safe_float(r.get("latency_s")),
                "target_tokens_per_sec": safe_float(r.get("tokens_per_sec")),
                "k_requested": safe_int(r.get("k"), k_actual),
                "temperature": safe_float(r.get("temperature"), 0.0),
                "prompt_token_len": safe_float(r.get("prompt_token_len"), 0.0),
                "n_output_tokens": safe_float(r.get("n_output_tokens"), 0.0),
                "num_turns": safe_int(r.get("num_turns"), 1),
                "max_prompt_tokens": safe_int(r.get("max_prompt_tokens"), 0),
            })

            # Causal block-history features.
            hist_n = len(past_accepted)
            outrow.update({
                "hist_n_blocks": hist_n,
                "hist_mean_accepted_len": float(np.mean(past_accepted)) if past_accepted else 0.0,
                "hist_std_accepted_len": float(np.std(past_accepted)) if past_accepted else 0.0,
                "hist_mean_first_rejection": float(np.mean(past_first_rej)) if past_first_rej else 0.0,
                "hist_full_accept_rate": float(np.mean(past_full)) if past_full else 0.0,
                "hist_reject_rate": 1.0 - float(np.mean(past_full)) if past_full else 0.0,
                "hist_mean_num_rejected": float(np.mean(past_num_rej)) if past_num_rej else 0.0,
            })

            for w in [1, 2, 4, 8, 16]:
                pa = past_accepted[-w:]
                pfw = past_full[-w:]
                pr = past_first_rej[-w:]
                nr = past_num_rej[-w:]

                outrow[f"hist_mean_accepted_len_last{w}"] = float(np.mean(pa)) if pa else 0.0
                outrow[f"hist_full_accept_rate_last{w}"] = float(np.mean(pfw)) if pfw else 0.0
                outrow[f"hist_mean_first_rejection_last{w}"] = float(np.mean(pr)) if pr else 0.0
                outrow[f"hist_mean_num_rejected_last{w}"] = float(np.mean(nr)) if nr else 0.0

            outrow.update(pf)

            # Anti-leakage audit fields.
            outrow["feature_cutoff_block"] = outrow["block_index"] - 1
            outrow["target_block"] = outrow["block_index"]
            outrow["uses_future_request_features"] = 0
            outrow["trace_found"] = int(tr is not None)

            rows.append(outrow)

            # Advance prefix position after this block.
            # vLLM speculative step advances by accepted draft tokens plus one verifier token.
            prefix_pos += accepted_len + 1
            if token_ids:
                prefix_pos = min(prefix_pos, len(token_ids))

            # Update history after target extraction.
            past_accepted.append(accepted_len)
            past_full.append(full_accept)
            past_first_rej.append(first_rej)
            past_num_rej.append(num_rej)

    outdf = pd.DataFrame(rows)
    outdf.to_csv(out, index=False)

    audit = {
        "input_block_rows": int(len(block)),
        "output_rows": int(len(outdf)),
        "rows_without_trace": int(n_no_trace),
        "trace_missing_rate": float(n_no_trace / max(1, n_total)),
        "core_only": bool(args.core_only),
        "forbidden_features_excluded": [
            "mean_entropy",
            "repetition_density",
            "entropy_bucket",
            "oracle_k_estimated",
            "oracle_expected_tokens",
            "final request-level aggregates as controller inputs",
        ],
        "causal_feature_rule": "Each block row uses prefix tokens before the current block and block outcomes strictly before the current block.",
        "prefix_position_rule": "prefix_pos_before_block_b = sum_{i < b}(accepted_len_i + 1), capped at n_output_tokens.",
    }

    with open(out.parent / "feature_leakage_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print("Wrote:", out)
    print("Rows:", outdf.shape)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
