import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                pass


def num(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def first_num(d, keys, default=None):
    for k in keys:
        if k in d:
            v = num(d.get(k), None)
            if v is not None:
                return v
    return default


def parse_mask(mask):
    """
    Return accepted_true, rejected_false, total_mask_len.
    Strictly uses accepted_mask only.
    """
    if mask is None:
        return None

    if not isinstance(mask, list):
        return None

    true_count = 0
    false_count = 0

    for x in mask:
        if x is True:
            true_count += 1
        elif x is False:
            false_count += 1
        elif isinstance(x, (int, float)):
            if int(x) == 1:
                true_count += 1
            elif int(x) == 0:
                false_count += 1
        elif isinstance(x, str):
            lx = x.strip().lower()
            if lx in {"true", "1", "t", "yes"}:
                true_count += 1
            elif lx in {"false", "0", "f", "no"}:
                false_count += 1

    return true_count, false_count, true_count + false_count


def load_hot_results(root, label):
    root = Path(root)
    rows = []

    for p in (root / "results").glob("*.json"):
        r = read_json(p)
        if not r:
            continue
        if r.get("status") not in {None, "SUCCESS", "success"}:
            continue

        runtime_s = first_num(r, ["wall_s", "latency_s", "runtime_s", "elapsed_s"])
        tps = first_num(r, ["tokens_per_sec", "tps", "throughput_tps"])
        out_tok = first_num(
            r,
            ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens"],
            0,
        )

        if runtime_s is None or tps is None:
            continue

        rows.append({
            "run_family": label,
            "system": f"{label}_{r.get('pool')}",
            "pool": r.get("pool"),
            "workload": r.get("workload"),
            "method": r.get("method"),
            "slow_k": r.get("slow_k"),
            "global_index": r.get("global_index"),
            "source_index": r.get("source_index"),
            "runtime_s": runtime_s,
            "tokens_per_sec": tps,
            "n_output_tokens": out_tok,
        })

    return pd.DataFrame(rows)


def aggregate_hot_masks(root, label):
    root = Path(root)

    # Keyed by pool/global_index so each block aggregate attaches to the right request.
    by_request = {}
    active_rows = []

    total_events = 0
    mask_events = 0
    missing_mask_events = 0

    for p in root.rglob("block_events.jsonl"):
        if any(part.startswith("gpu") for part in p.parts):
            continue

        for e in read_jsonl(p):
            total_events += 1

            pool = e.get("pool")
            gi = e.get("global_index")

            if pool is None or gi is None:
                continue

            parsed = parse_mask(e.get("accepted_mask"))
            if parsed is None:
                missing_mask_events += 1
                continue

            mask_events += 1
            accepted_true, rejected_false, mask_len = parsed

            key = (str(pool), str(gi))
            d = by_request.setdefault(key, {
                "draft_tokens": 0.0,
                "accepted_tokens_total": 0.0,
                "rejected_tokens": 0.0,
                "n_blocks": 0,
                "mask_events": 0,
                "missing_mask_events": 0,
            })

            d["draft_tokens"] += mask_len
            d["accepted_tokens_total"] += accepted_true
            d["rejected_tokens"] += rejected_false
            d["n_blocks"] += 1
            d["mask_events"] += 1

            active_rows.append({
                "run_family": label,
                "pool": pool,
                "workload": e.get("workload"),
                "method": e.get("method"),
                "slow_k": e.get("slow_k"),
                "active_k": e.get("active_k", e.get("k_actual")),
                "mask_len": mask_len,
                "accepted_true": accepted_true,
                "rejected_false": rejected_false,
            })

    diagnostics = {
        "label": label,
        "total_block_events": total_events,
        "mask_events": mask_events,
        "missing_mask_events": missing_mask_events,
    }

    return by_request, pd.DataFrame(active_rows), diagnostics


def attach_masks_to_hot(hot_df, mask_agg):
    if hot_df.empty:
        return hot_df

    rows = []

    for _, r in hot_df.iterrows():
        rd = r.to_dict()
        key = (str(r["pool"]), str(r["global_index"]))

        a = mask_agg.get(key, {
            "draft_tokens": 0.0,
            "accepted_tokens_total": 0.0,
            "rejected_tokens": 0.0,
            "n_blocks": 0,
            "mask_events": 0,
            "missing_mask_events": 0,
        })

        rd.update(a)
        rows.append(rd)

    return pd.DataFrame(rows)


def load_baselines(root, hot_all):
    """
    Baselines are still read from traces.jsonl because they are not part of the
    hot block_events control traces. Their existing rejected_tokens/draft_tokens
    are used.
    """
    root = Path(root)

    needed = set()
    for _, r in hot_all.iterrows():
        if pd.notna(r.get("workload")) and pd.notna(r.get("source_index")):
            try:
                needed.add((str(r["workload"]), int(r["source_index"])))
            except Exception:
                pass

    rows = []

    for p in root.rglob("traces.jsonl"):
        for r in read_jsonl(p):
            workload = r.get("workload")
            source_index = r.get("request_ordinal", r.get("source_index"))
            method = r.get("method")

            if workload is None or source_index is None or method is None:
                continue

            try:
                key = (str(workload), int(source_index))
            except Exception:
                continue

            if needed and key not in needed:
                continue

            k = r.get("k", r.get("num_speculative_tokens", "NA"))
            system = "baseline_ar" if method == "ar" else f"baseline_{method}_k{k}"

            runtime_s = first_num(r, ["latency_s", "wall_s", "runtime_s", "elapsed_s"])
            tps = first_num(r, ["tokens_per_sec", "tps", "throughput_tps"])
            out_tok = first_num(
                r,
                ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens"],
                0,
            )

            if runtime_s is None or tps is None:
                continue

            draft = first_num(r, ["draft_tokens", "num_draft_tokens"], 0) or 0
            rejected = first_num(r, ["rejected_tokens", "num_rejected_tokens"], 0) or 0

            rows.append({
                "run_family": "baseline",
                "system": system,
                "pool": "baseline",
                "workload": workload,
                "method": method,
                "slow_k": k,
                "global_index": None,
                "source_index": int(source_index),
                "runtime_s": runtime_s,
                "tokens_per_sec": tps,
                "n_output_tokens": out_tok,
                "draft_tokens": draft,
                "accepted_tokens_total": max(draft - rejected, 0),
                "rejected_tokens": rejected,
                "n_blocks": 0,
                "mask_events": 0,
                "missing_mask_events": 0,
            })

    return pd.DataFrame(rows)


def summarize(df, group_cols):
    s = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_requests=("runtime_s", "count"),
            total_runtime_s=("runtime_s", "sum"),
            avg_runtime_s=("runtime_s", "mean"),
            mean_tps=("tokens_per_sec", "mean"),
            draft_tokens_total=("draft_tokens", "sum"),
            accepted_tokens_total=("accepted_tokens_total", "sum"),
            rejected_tokens_total=("rejected_tokens", "sum"),
            n_blocks_total=("n_blocks", "sum"),
            mask_events_total=("mask_events", "sum"),
            missing_mask_events_total=("missing_mask_events", "sum"),
        )
        .reset_index()
    )

    s["wasted_token_pct"] = (
        100 * s["rejected_tokens_total"] / s["draft_tokens_total"].replace({0: pd.NA})
    )
    s.loc[s["system"].eq("baseline_ar"), "wasted_token_pct"] = pd.NA

    s["accepted_per_block"] = (
        s["accepted_tokens_total"] / s["n_blocks_total"].replace({0: pd.NA})
    )

    return s


def add_speedups(s):
    if "workload" in s.columns:
        ar = s[s["system"] == "baseline_ar"][["workload", "mean_tps", "avg_runtime_s"]].rename(
            columns={"mean_tps": "ar_mean_tps", "avg_runtime_s": "ar_avg_runtime_s"}
        )
        out = s.merge(ar, on="workload", how="left")
    else:
        out = s.copy()
        ar = s[s["system"] == "baseline_ar"]
        if len(ar):
            out["ar_mean_tps"] = ar.iloc[0]["mean_tps"]
            out["ar_avg_runtime_s"] = ar.iloc[0]["avg_runtime_s"]

    out["mean_tps_speedup_vs_ar"] = out["mean_tps"] / out["ar_mean_tps"]
    out["latency_speedup_vs_ar"] = out["ar_avg_runtime_s"] / out["avg_runtime_s"]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hot", nargs=2, action="append", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hot_dfs = []
    active_dfs = []
    diagnostics = []

    for label, root in args.hot:
        print(f"[hot] {label}: {root}")

        h = load_hot_results(root, label)
        mask_agg, active, diag = aggregate_hot_masks(root, label)
        h = attach_masks_to_hot(h, mask_agg)

        print("  hot rows:", len(h))
        print("  block events:", diag["total_block_events"])
        print("  mask events:", diag["mask_events"])
        print("  missing mask events:", diag["missing_mask_events"])

        if len(h):
            hot_dfs.append(h)
        if len(active):
            active_dfs.append(active)

        diagnostics.append(diag)

    hot_all = pd.concat(hot_dfs, ignore_index=True) if hot_dfs else pd.DataFrame()
    active_all = pd.concat(active_dfs, ignore_index=True) if active_dfs else pd.DataFrame()

    baseline = load_baselines(args.baseline_root, hot_all)
    print("[baseline rows]", len(baseline))

    all_df = pd.concat([baseline, hot_all], ignore_index=True)

    all_df.to_csv(out_dir / "request_level_all_ablation_and_baselines_MASK.csv", index=False)
    active_all.to_csv(out_dir / "active_k_all_ablations_raw_MASK.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "accepted_mask_diagnostics.csv", index=False)

    overall = add_speedups(summarize(all_df, ["run_family", "system", "pool"]))
    overall = overall.sort_values("mean_tps_speedup_vs_ar", ascending=False)
    overall.to_csv(out_dir / "summary_overall_all_ablation_and_baselines_MASK.csv", index=False)

    by_workload = add_speedups(summarize(all_df, ["run_family", "system", "pool", "workload"]))
    by_workload = by_workload.sort_values(["workload", "mean_tps_speedup_vs_ar"], ascending=[True, False])
    by_workload.to_csv(out_dir / "summary_by_workload_all_ablation_and_baselines_MASK.csv", index=False)

    cols = [
        "system", "n_requests", "mean_tps_speedup_vs_ar", "latency_speedup_vs_ar",
        "wasted_token_pct", "accepted_per_block", "n_blocks_total",
        "mask_events_total", "missing_mask_events_total",
        "avg_runtime_s", "mean_tps",
        "draft_tokens_total", "rejected_tokens_total",
    ]

    print("\n=== OVERALL USING accepted_mask ===")
    print(overall[cols].round(3).to_string(index=False))

    print("\nwrote:")
    for p in out_dir.glob("*MASK*.csv"):
        print(p)
    print(out_dir / "accepted_mask_diagnostics.csv")


if __name__ == "__main__":
    main()
