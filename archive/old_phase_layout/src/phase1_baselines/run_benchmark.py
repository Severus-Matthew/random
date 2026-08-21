#!/usr/bin/env python
import argparse, json, time, math, os, random
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── vLLM speculative config ────────────────────────────────────────────────
def build_speculative_config(method: str, k: int, draft_model: str = None, eagle3_model: str = None) -> Dict[str, Any] | None:
    """
    vLLM speculative configs. If your vLLM version uses different keys,
    update only this function.
    """
    if method == "ar":
        return None
    if method == "ngram_sd":
        return {
            "method": "ngram",
            "num_speculative_tokens": k,
            "prompt_lookup_max": max(2, k),
            "prompt_lookup_min": 1,
        }
    if method == "draft_sd":
        if not draft_model:
            raise ValueError("draft_sd requires --draft_model")
        return {
            "method": "draft_model",
            "model": draft_model,
            "num_speculative_tokens": k,
        }
    if method == "eagle3":
        if not eagle3_model:
            raise ValueError("eagle3 requires --eagle3_model")
        return {
            "method": "eagle3",
            "model": eagle3_model,
            "num_speculative_tokens": k,
        }
    raise ValueError(f"Unknown method: {method}")


# ── Entropy proxy from top-k logprobs ─────────────────────────────────────
def approx_entropy_from_logprobs(step_logprobs) -> float:
    """
    Approximate entropy from vLLM's top-k logprobs.
    Lower bound because only top-k returned, but sufficient for regime analysis.
    """
    if not step_logprobs:
        return float("nan")
    vals = []
    for lp in step_logprobs.values():
        if hasattr(lp, "logprob"):
            vals.append(lp.logprob)
        elif isinstance(lp, dict) and "logprob" in lp:
            vals.append(lp["logprob"])
        elif isinstance(lp, (float, int)):
            vals.append(float(lp))
    if not vals:
        return float("nan")
    probs = np.exp(np.array(vals) - np.max(vals))
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


# ── Structural metrics ─────────────────────────────────────────────────────
def compute_repetition_density(token_ids: List[int], window: int = 32) -> float:
    """
    Fraction of tokens that are exact repetitions within a rolling window.
    High value → highly repetitive / structured region (n-gram-friendly).
    """
    if len(token_ids) < 2:
        return 0.0
    hits = 0
    for i in range(1, len(token_ids)):
        lookback = token_ids[max(0, i - window): i]
        if token_ids[i] in lookback:
            hits += 1
    return hits / (len(token_ids) - 1)


def compute_acceptance_volatility(accepted: List[float]) -> float:
    """
    Std-dev of rolling acceptance signal.
    High volatility → unstable speculative regime (bad for fixed-k policies).
    NaN-safe.
    """
    clean = [x for x in accepted if not (isinstance(x, float) and math.isnan(x))]
    if len(clean) < 2:
        return float("nan")
    return float(np.std(clean))


def compute_rejection_locality(rejected: List[int], window: int = 16) -> float:
    """
    Fraction of rejection events that are clustered (another rejection within window).
    High value → rejections come in bursts (structural boundary).
    """
    if not rejected or sum(rejected) == 0:
        return float("nan")
    reject_positions = [i for i, r in enumerate(rejected) if r]
    if len(reject_positions) < 2:
        return 0.0
    clustered = 0
    for idx, pos in enumerate(reject_positions):
        neighbors = reject_positions[max(0, idx - 1): idx] + reject_positions[idx + 1: idx + 2]
        if any(abs(pos - n) <= window for n in neighbors):
            clustered += 1
    return clustered / len(reject_positions)


def compute_verifier_utilization(n_output_tokens: int, k: int, method: str) -> float:
    """
    Fraction of wall-clock compute allocated to the verifier model.
    For AR: 1.0 (all tokens go through verifier).
    For speculative methods: estimated as verifier_passes / total_possible_passes.
    """
    if method == "ar" or k <= 0:
        return 1.0
    verifier_passes = math.ceil(n_output_tokens / max(1, k))
    max_possible_passes = n_output_tokens  # if every token needed its own pass
    return verifier_passes / max(1, max_possible_passes)


def compute_rollback_frequency(rejected: List[int], k: int) -> float:
    """
    Estimated rollback events per verifier call.
    A rollback = at least one rejection in a speculation window.
    Proxy: count windows that had any rejection.
    """
    if not rejected or k <= 0:
        return float("nan")
    n_windows = math.ceil(len(rejected) / max(1, k))
    if n_windows == 0:
        return float("nan")
    rollbacks = 0
    for w in range(n_windows):
        window_slice = rejected[w * k: (w + 1) * k]
        if any(window_slice):
            rollbacks += 1
    return rollbacks / n_windows


# ── Token trace extraction ─────────────────────────────────────────────────
def extract_token_trace(output, tokenizer, latency_s: float, method: str, k: int) -> dict:
    text = output.outputs[0].text
    token_ids = output.outputs[0].token_ids
    n_out = len(token_ids)

    logprobs = getattr(output.outputs[0], "logprobs", None) or []
    entropies = [
        approx_entropy_from_logprobs(logprobs[i]) if i < len(logprobs) else float("nan")
        for i in range(n_out)
    ]

    # vLLM per-token acceptance internals are version-dependent.
    # We compute what we can; Phase 3+ can plug in richer vLLM hooks.
    accepted = [float("nan")] * n_out
    rejected = [0] * n_out

    # ── Structural metrics ──
    rep_density = compute_repetition_density(list(map(int, token_ids)))
    acc_volatility = compute_acceptance_volatility(accepted)
    rej_locality = compute_rejection_locality(rejected)
    verifier_util = compute_verifier_utilization(n_out, k, method)
    rollback_freq = compute_rollback_frequency(rejected, k)

    return {
        "text": text,
        "n_output_tokens": n_out,
        "token_ids": list(map(int, token_ids)),
        "token_strings": [tokenizer.decode([int(t)]) for t in token_ids],
        "entropy": entropies,
        "accepted": accepted,
        "rejected": rejected,
        "latency_s": latency_s,
        "tokens_per_sec": n_out / max(latency_s, 1e-9),
        "method": method,
        "k": k,
        # structural
        "repetition_density": rep_density,
        "acceptance_volatility": acc_volatility,
        "rejection_locality": rej_locality,
        "verifier_utilization": verifier_util,
        "rollback_frequency": rollback_freq,
    }




# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload",    required=True)
    ap.add_argument("--run_name",    default=None, help="Output folder name; useful when one workload has multiple jsonl files")
    ap.add_argument("--limit",       type=int, default=0, help="If >0, cap number of jsonl rows read")
    ap.add_argument("--method",      required=True, choices=["ar", "ngram_sd", "draft_sd", "eagle3"])
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--model",       default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--draft_model", default=os.environ.get("DRAFT_MODEL",  "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--eagle3_model",default=os.environ.get("EAGLE3_MODEL", "RedHatAI/Qwen3-8B-speculator.eagle3"))
    ap.add_argument("--k",           type=int,   default=4)
    ap.add_argument("--max_tokens",  type=int,   default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_logprobs",type=int,   default=5)
    ap.add_argument("--tensor_parallel_size", type=int, default=int(os.environ.get("TP", "1")))
    ap.add_argument("--seed",        type=int,   default=int(os.environ.get("SEED", "42")))
    args = ap.parse_args()

    # ── Seed everything ──
    set_seed(args.seed)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    run_name = args.run_name or args.workload
    out_dir = Path(args.out_dir) / run_name / args.method / f"k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config snapshot for reproducibility ──
    config_snapshot = {
        "workload":    args.workload,
        "method":      args.method,
        "k":           args.k,
        "model":       args.model,
        "draft_model": args.draft_model,
        "eagle3_model":args.eagle3_model,
        "max_tokens":  args.max_tokens,
        "temperature": args.temperature,
        "top_logprobs":args.top_logprobs,
        "seed":        args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "input_jsonl": args.input_jsonl,
        "run_name":    run_name,
        "limit":       args.limit,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)
    print("[CONFIG]", json.dumps(config_snapshot))

    prompts = [json.loads(l) for l in open(args.input_jsonl)]
    if args.limit and args.limit > 0:
        prompts = prompts[:args.limit]
    print(f"[DATA] Loaded {len(prompts)} rows from {args.input_jsonl}")
    spec_config = build_speculative_config(args.method, args.k, args.draft_model, args.eagle3_model)

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    if spec_config is not None:
        llm_kwargs["speculative_config"] = spec_config

    print("[INFO] Loading LLM:", llm_kwargs)
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        logprobs=args.top_logprobs,
        seed=args.seed,
    )

    rows = []
    trace_path = out_dir / "traces.jsonl"
    with trace_path.open("w") as tf:
        for row in tqdm(prompts):
            prompt = row["prompt"]
            t0 = time.time()
            outputs = llm.generate([prompt], sp, use_tqdm=False)
            latency = time.time() - t0
            out = outputs[0]
            trace = extract_token_trace(out, tokenizer, latency, args.method, args.k)

            n = trace["n_output_tokens"]
            accepted_known = [x for x in trace["accepted"] if not (isinstance(x, float) and math.isnan(x))]
            acceptance_rate     = float(np.mean(accepted_known)) if accepted_known else float("nan")
            accepted_tokens     = float(np.nansum(trace["accepted"])) if accepted_known else float("nan")
            verifier_passes     = max(1, math.ceil(n / max(1, args.k)))
            accepted_per_verifier = accepted_tokens / verifier_passes if accepted_known else float("nan")

            rec = {
                "id":             row.get("id", row.get("uid", f"{args.workload}_{len(rows)}")),
                "workload":       args.workload,
                "source_dataset": row.get("source_dataset"),
                "method":         args.method,
                "k":              args.k,
                "seed":           args.seed,
                # core inference metrics
                "n_output_tokens":                  n,
                "latency_s":                        latency,
                "tokens_per_sec":                   trace["tokens_per_sec"],
                "acceptance_rate":                  acceptance_rate,
                "accepted_tokens_per_verifier_pass":accepted_per_verifier,
                # Phase 1 required metrics
                "rollback_frequency":               trace["rollback_frequency"],
                "verifier_utilization":             trace["verifier_utilization"],
                # structural metrics
                "mean_entropy":          float(np.nanmean(trace["entropy"])) if trace["entropy"] else float("nan"),
                "repetition_density":    trace["repetition_density"],
                "acceptance_volatility": trace["acceptance_volatility"],
                "rejection_locality":    trace["rejection_locality"],
            }
            rows.append(rec)
            tf.write(json.dumps({**rec, **trace}) + "\n")

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
