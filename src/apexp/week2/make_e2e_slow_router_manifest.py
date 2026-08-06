from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Could not find any of columns: {candidates}\\nAvailable: {list(df.columns)}")
    return None


def sanitize(x: str) -> str:
    return (
        str(x)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow-router-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens-default", type=int, default=512)
    args = ap.parse_args()

    slow_dir = Path(args.slow_router_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = slow_dir / "test_candidates_scored.csv"
    if not cand_path.exists():
        raise SystemExit(f"Missing {cand_path}")

    df = pd.read_csv(cand_path)

    prompt_col = pick_col(df, ["prompt_hash", "prompt_id"])
    text_col = pick_col(df, ["prompt_text", "prompt", "input_text"])
    workload_col = pick_col(df, ["workload"])
    method_col = pick_col(df, ["method"])
    k_col = pick_col(df, ["k_requested", "k"])
    temp_col = pick_col(df, ["temperature"], required=False)

    score_col = pick_col(
        df,
        [
            "pred_tps",
            "pred_actual_tps",
            "predicted_tps",
            "router_score",
            "score",
            "pred",
        ],
    )

    # Pick one slow-router action per prompt.
    idx = df.groupby(prompt_col)[score_col].idxmax()
    sel = df.loc[idx].copy()

    # Deterministic optional subsample for quick sanity runs.
    sel = sel.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_prompts and args.max_prompts > 0:
        sel = sel.head(args.max_prompts).copy()

    selected_path = out_dir / "selected_slow_router_actions.csv"
    sel.to_csv(selected_path, index=False)

    input_dir = out_dir / "inputs_by_action"
    input_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    # Group by workload/method/k/temp so each vLLM run uses one action.
    group_cols = [workload_col, method_col, k_col]
    if temp_col:
        group_cols.append(temp_col)

    for gid, (key, g) in enumerate(sel.groupby(group_cols, dropna=False)):
        if not isinstance(key, tuple):
            key = (key,)

        values = dict(zip(group_cols, key))
        workload = str(values[workload_col])
        method = str(values[method_col])
        k = int(values[k_col])
        temp = float(values[temp_col]) if temp_col else 0.0

        input_path = input_dir / f"group_{gid:04d}_{sanitize(workload)}_{sanitize(method)}_k{k}_temp{temp}.jsonl"

        with open(input_path, "w") as f:
            for _, r in g.iterrows():
                obj = {
                    "prompt": str(r[text_col]),
                    "prompt_text": str(r[text_col]),
                    "prompt_hash": str(r[prompt_col]),
                    "prompt_id": str(r[prompt_col]),
                    "workload": workload,
                }
                f.write(json.dumps(obj) + "\n")

        max_tokens = args.max_tokens_default
        if "max_tokens" in g.columns and pd.notna(g["max_tokens"].iloc[0]):
            try:
                max_tokens = int(g["max_tokens"].iloc[0])
            except Exception:
                pass

        draft_model = "NA"
        eagle3_model = "NA"
        ngram_min = "NA"
        ngram_max = "NA"

        if method == "draft_sd":
            draft_model = "Qwen/Qwen2.5-1.5B-Instruct"
        elif method == "eagle3":
            eagle3_model = "RedHatAI/Qwen3-8B-speculator.eagle3"
        elif method == "ngram_sd":
            ngram_min = "1"
            ngram_max = "4"

        job_id = f"e2e_{gid:04d}_{sanitize(workload)}_{sanitize(method)}_k{k}"

        run_dir = out_dir / "runs" / job_id

        manifest_rows.append({
            "priority": gid,
            "job_id": job_id,
            "source_experiments": "e2e_slow_router",
            "phase": 2,
            "workload": workload,
            "input_jsonl": str(input_path),
            "method": method,
            "k": k,
            "temperature": temp,
            "ngram_lookup_min": ngram_min,
            "ngram_lookup_max": ngram_max,
            "draft_model": draft_model,
            "eagle3_model": eagle3_model,
            "target_model": "Qwen/Qwen3-8B",
            "max_prompt_tokens": 0,
            "num_turns": 1,
            "limit": len(g),
            "max_tokens": max_tokens,
            "seed": args.seed,
            "run_dir": str(run_dir),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    print("selected prompts:", len(sel))
    print("groups:", len(manifest))
    print("wrote:", selected_path)
    print("wrote:", manifest_path)
    print(manifest[["job_id", "workload", "method", "k", "limit"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
