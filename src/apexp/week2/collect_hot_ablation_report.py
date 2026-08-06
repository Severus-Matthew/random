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


def load_hot(root, label):
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
        out_tok = first_num(r, ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens"], 0)

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

    df = pd.DataFrame(rows)

    # block-level waste, keyed by pool/global_index
    agg = {}
    active_rows = []

    for p in root.rglob("block_events.jsonl"):
        if any(part.startswith("gpu") for part in p.parts):
            continue

        for e in read_jsonl(p):
            pool = e.get("pool")
            gi = e.get("global_index")
            if pool is None or gi is None:
                continue

            k_actual = first_num(e, ["k_actual", "scheduled_spec_len", "active_k"], 0)
            active_k = first_num(e, ["active_k", "k_actual", "scheduled_spec_len"], k_actual)
            accepted = first_num(e, ["accepted_len", "num_accepted_tokens"], 0)
            rejected = first_num(e, ["num_rejected", "rejected_tokens"], None)

            if rejected is None:
                rejected = max(k_actual - accepted, 0)

            draft = accepted + rejected

            key = (str(pool), str(gi))
            d = agg.setdefault(key, {
                "draft_tokens": 0.0,
                "accepted_tokens_total": 0.0,
                "rejected_tokens": 0.0,
                "n_blocks": 0,
            })
            d["draft_tokens"] += draft
            d["accepted_tokens_total"] += accepted
            d["rejected_tokens"] += rejected
            d["n_blocks"] += 1

            active_rows.append({
                "run_family": label,
                "pool": pool,
                "workload": e.get("workload"),
                "method": e.get("method"),
                "slow_k": e.get("slow_k"),
                "active_k": active_k,
                "accepted_len": accepted,
                "num_rejected": rejected,
            })

    if df.empty:
        return df, pd.DataFrame(active_rows)

    out_rows = []
    for _, r in df.iterrows():
        rd = r.to_dict()
        key = (str(r["pool"]), str(r["global_index"]))
        a = agg.get(key, {
            "draft_tokens": 0.0,
            "accepted_tokens_total": 0.0,
            "rejected_tokens": 0.0,
            "n_blocks": 0,
        })
        rd.update(a)
        out_rows.append(rd)

    return pd.DataFrame(out_rows), pd.DataFrame(active_rows)


def load_baselines(root, hot_all):
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
            out_tok = first_num(r, ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens"], 0)

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
        )
        .reset_index()
    )

    s["wasted_token_pct"] = 100 * s["rejected_tokens_total"] / s["draft_tokens_total"].replace({0: pd.NA})
    s.loc[s["system"].eq("baseline_ar"), "wasted_token_pct"] = pd.NA
    s["accepted_per_block"] = s["accepted_tokens_total"] / s["n_blocks_total"].replace({0: pd.NA})
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

    for label, root in args.hot:
        print(f"[hot] {label}: {root}")
        h, a = load_hot(root, label)
        print("  hot rows:", len(h), "active rows:", len(a))
        if len(h):
            hot_dfs.append(h)
        if len(a):
            active_dfs.append(a)

    hot_all = pd.concat(hot_dfs, ignore_index=True) if hot_dfs else pd.DataFrame()
    active_all = pd.concat(active_dfs, ignore_index=True) if active_dfs else pd.DataFrame()

    baseline = load_baselines(args.baseline_root, hot_all)
    print("[baseline rows]", len(baseline))

    all_df = pd.concat([baseline, hot_all], ignore_index=True)

    all_df.to_csv(out_dir / "request_level_all_ablation_and_baselines.csv", index=False)
    active_all.to_csv(out_dir / "active_k_all_ablations_raw.csv", index=False)

    overall = add_speedups(summarize(all_df, ["run_family", "system", "pool"]))
    overall = overall.sort_values("mean_tps_speedup_vs_ar", ascending=False)
    overall.to_csv(out_dir / "summary_overall_all_ablation_and_baselines.csv", index=False)

    by_workload = add_speedups(summarize(all_df, ["run_family", "system", "pool", "workload"]))
    by_workload = by_workload.sort_values(["workload", "mean_tps_speedup_vs_ar"], ascending=[True, False])
    by_workload.to_csv(out_dir / "summary_by_workload_all_ablation_and_baselines.csv", index=False)

    print("\n=== OVERALL ===")
    cols = [
        "system", "n_requests", "mean_tps_speedup_vs_ar", "latency_speedup_vs_ar",
        "wasted_token_pct", "accepted_per_block", "n_blocks_total",
        "avg_runtime_s", "mean_tps"
    ]
    print(overall[cols].round(3).to_string(index=False))

    print("\nwrote:")
    for p in out_dir.glob("*.csv"):
        print(p)


if __name__ == "__main__":
    main()
