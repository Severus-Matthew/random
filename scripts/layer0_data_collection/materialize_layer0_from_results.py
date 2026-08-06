#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def to_float(x, default=None):
    if x is None:
        return default
    try:
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return default
        return float(s)
    except Exception:
        return default


def to_int(x, default=None):
    v = to_float(x, None)
    if v is None:
        return default
    return int(v)


def clamp01(x):
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def collect_survival(row, max_depth):
    """
    Interpret vLLM positional acceptance counters as survival-prefix estimates:
        accept_rate_pos_j = S_j = P(positions 0..j accepted)
    """
    survival = []

    for j in range(max_depth):
        val = None
        for key in (
            f"survival_pos_{j}",
            f"accept_rate_pos_{j}",
            f"accept_rate_{j}",
            f"pos_accept_rate_{j}",
        ):
            if key in row:
                val = to_float(row.get(key), None)
                if val is not None:
                    break

        if val is None:
            break

        survival.append(clamp01(val))

    fixed = []
    prev = 1.0
    violations = 0

    for s in survival:
        if s is None:
            break
        if s > prev + 1e-8:
            violations += 1
            s = prev
        fixed.append(s)
        prev = s

    return fixed, violations


def derive_survival_quantities(survival):
    out = {}
    prev = 1.0
    observed_expected = 0.0

    for j, s in enumerate(survival):
        observed_expected += s

        if j == 0:
            conditional_accept = s
            pmf = 1.0 - s
        else:
            conditional_accept = s / prev if prev > 1e-12 else 0.0
            pmf = prev - s

        hazard = 1.0 - conditional_accept

        out[f"survival_pos_{j}"] = s
        out[f"conditional_accept_pos_{j}"] = clamp01(conditional_accept)
        out[f"hazard_pos_{j}"] = clamp01(hazard)
        out[f"pmf_L_eq_{j}"] = max(0.0, pmf)

        prev = s

    if survival:
        pmf_prefix = sum(out[f"pmf_L_eq_{j}"] for j in range(len(survival)))
        out["apex_observed_tail_mass_L_ge_n"] = prev
        out["apex_pmf_prefix_plus_tail"] = pmf_prefix + prev
        out["apex_expected_len_survival_observed"] = observed_expected
    else:
        out["apex_observed_tail_mass_L_ge_n"] = None
        out["apex_pmf_prefix_plus_tail"] = None
        out["apex_expected_len_survival_observed"] = None

    return out


def read_config(run_dir):
    cfg_path = run_dir / "config.json"
    print(run_dir)
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def infer_from_path(run_dir):
    parts = list(run_dir.parts)
    meta = {
        "experiment": None,
        "run_name": None,
        "method": None,
        "run_leaf": run_dir.name,
    }

    # Expected pattern:
    # results / EXPERIMENT / RUN_NAME / METHOD / LEAF
    if len(parts) >= 4:
        meta["method"] = parts[-2]
        meta["run_name"] = parts[-3]
        meta["experiment"] = parts[-4]

    return meta


def materialize_one_run(run_dir, max_depth, overwrite=False):
    summary_path = run_dir / "summary.csv"
    out_path = run_dir / "apex_layer0_request.jsonl"

    if not summary_path.exists():
        return 0, f"skip missing summary.csv: {run_dir}"

    if out_path.exists() and not overwrite:
        return 0, f"skip exists: {out_path}"

    cfg = read_config(run_dir)
    path_meta = infer_from_path(run_dir)

    rows_written = 0

    with summary_path.open("r", newline="") as f, out_path.open("w") as out:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            survival, violations = collect_survival(row, max_depth=max_depth)
            q = derive_survival_quantities(survival)

            accepted_tokens_total = to_float(row.get("accepted_tokens_total"), None)
            draft_tokens = to_float(row.get("draft_tokens"), None)
            accepted_per_draft = to_float(row.get("accepted_per_draft"), None)

            expected_counter = accepted_per_draft
            if expected_counter is None and accepted_tokens_total is not None and draft_tokens and draft_tokens > 0:
                expected_counter = accepted_tokens_total / draft_tokens

            observed_expected = q.get("apex_expected_len_survival_observed")
            tail_missing = None
            if expected_counter is not None and observed_expected is not None:
                tail_missing = expected_counter - observed_expected

            trace = {
                "apex_trace_level": "request_aggregate",
                "apex_strict_block_trace": False,
                "apex_note": (
                    "Materialized from summary.csv positional counters. "
                    "This is request-level survival logging, not exact per-block first-rejection logging."
                ),

                "request_index": i,
                "request_id": row.get("request_id") or row.get("id") or str(i),

                "experiment": row.get("experiment") or cfg.get("experiment") or path_meta.get("experiment"),
                "run_name": row.get("run_name") or cfg.get("run_name") or path_meta.get("run_name"),
                "workload": row.get("workload") or cfg.get("workload"),
                "method": row.get("method") or cfg.get("method") or path_meta.get("method"),

                "model": row.get("model") or cfg.get("model"),
                "draft_model": row.get("draft_model") or cfg.get("draft_model") or cfg.get("eagle3_model"),
                "eagle3_model": row.get("eagle3_model") or cfg.get("eagle3_model"),

                "k": to_int(row.get("k") or cfg.get("k"), None),
                "temperature": to_float(row.get("temperature") or cfg.get("temperature"), None),

                "latency_s": to_float(row.get("latency_s"), None),
                "tokens_per_sec": to_float(row.get("tokens_per_sec"), None),
                "output_tokens": to_float(row.get("output_tokens"), None),

                "draft_tokens": draft_tokens,
                "accepted_tokens_total": accepted_tokens_total,
                "accepted_per_draft": accepted_per_draft,
                "acceptance_rate": to_float(row.get("acceptance_rate") or row.get("accepted_per_draft"), None),

                "mean_entropy": to_float(row.get("mean_entropy"), None),
                "repetition_density": to_float(row.get("repetition_density"), None),

                "oracle_k_estimated": to_float(row.get("oracle_k_estimated"), None),
                "oracle_expected_tokens": to_float(row.get("oracle_expected_tokens"), None),

                "apex_observed_positions": len(survival),
                "apex_survival_monotonic_violations": violations,
                "apex_expected_len_counter": expected_counter,
                "apex_expected_len_survival_observed": observed_expected,
                "apex_tail_expected_missing": tail_missing,

                "metadata": {
                    "source_summary_csv": str(summary_path),
                    "source_run_dir": str(run_dir),
                },
            }

            trace.update(q)
            out.write(json.dumps(trace) + "\n")
            rows_written += 1

    return rows_written, f"wrote {rows_written}: {out_path}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_dir)
    summary_files = sorted(root.rglob("summary.csv"))

    if not summary_files:
        print(f"No summary.csv files found under {root}")
        return

    total_rows = 0

    for summary_path in summary_files:
        run_dir = summary_path.parent
        n, msg = materialize_one_run(
            run_dir=run_dir,
            max_depth=args.max_depth,
            overwrite=args.overwrite,
        )
        total_rows += n
        print(msg)

    print(f"\nLayer 0 materialization complete: files={len(summary_files)} rows_written={total_rows}")


if __name__ == "__main__":
    main()
