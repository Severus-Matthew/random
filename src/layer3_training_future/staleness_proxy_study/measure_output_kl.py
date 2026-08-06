# #!/usr/bin/env python
# """
# measure_output_kl.py — output-space drift distance (optional, more meaningful)
# ==============================================================================
# Weight-space distance is cheap but not directly interpretable. The quantity
# that actually governs SD acceptance is how much the target's *output
# distribution* moved. This script measures, on a fixed probe set, the mean
# per-token KL( p_drifted || p_ref ) between each drifted checkpoint and the
# reference. Acceptance-vs-KL is the scientifically clean x-axis for the paper.

# Runs a plain HF forward pass (no vLLM) so it is independent of the SD harness.
# Reference logits are computed once and cached; each drifted model is then
# streamed past it. Uses teacher-forcing on the probe prompts' own continuations.

# Example:
#   python measure_output_kl.py \
#       --ref_model Qwen/Qwen3-8B \
#       --manifest /scratch/drifted/drift_manifest.csv \
#       --probe_jsonl data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl \
#       --n_probe 64 --max_len 256 --out /scratch/drifted/drift_kl.csv
# """
# import argparse
# import csv
# import json
# from pathlib import Path

# import torch
# import torch.nn.functional as F
# from transformers import AutoModelForCausalLM, AutoTokenizer


# def _prompt(row):
#     for k in ("prompt", "input", "question", "problem_statement", "text", "content"):
#         if row.get(k):
#             return row[k]
#     return ""


# @torch.no_grad()
# def logits_for(model, input_ids, attn):
#     return model(input_ids=input_ids, attention_mask=attn).logits  # [B,T,V]


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ref_model", required=True)
#     ap.add_argument("--manifest", required=True, help="drift_manifest.csv from make_drifted_models")
#     ap.add_argument("--probe_jsonl", required=True)
#     ap.add_argument("--n_probe", type=int, default=64)
#     ap.add_argument("--max_len", type=int, default=256)
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--device", default="cuda")
#     ap.add_argument("--dtype", default="bfloat16")
#     ap.add_argument("--out", required=True)
#     args = ap.parse_args()

#     dtype = getattr(torch, args.dtype)
#     tok = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token

#     rows = [json.loads(l) for l in open(args.probe_jsonl)][: args.n_probe]
#     texts = [_prompt(r) for r in rows if _prompt(r)]
#     enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
#               max_length=args.max_len)
#     ids, attn = enc["input_ids"], enc["attention_mask"]

#     def batches():
#         for i in range(0, len(ids), args.batch_size):
#             yield ids[i:i + args.batch_size], attn[i:i + args.batch_size]

#     # 1) cache reference log-probs
#     print(f"[ref] {args.ref_model}")
#     ref = AutoModelForCausalLM.from_pretrained(
#         args.ref_model, torch_dtype=dtype, trust_remote_code=True).to(args.device).eval()
#     ref_logp = []
#     for b_ids, b_attn in batches():
#         lg = logits_for(ref, b_ids.to(args.device), b_attn.to(args.device))
#         ref_logp.append(F.log_softmax(lg.float(), dim=-1).cpu())
#     del ref
#     torch.cuda.empty_cache()

#     # 2) per drifted checkpoint: mean token KL(drifted || ref) over valid positions
#     man = list(csv.DictReader(open(args.manifest)))
#     out_rows = []
#     for m in man:
#         path = m["path"]
#         print(f"[kl ] {m['tag']}  ({path})")
#         try:
#             drift = AutoModelForCausalLM.from_pretrained(
#                 path, torch_dtype=dtype, trust_remote_code=True).to(args.device).eval()
#         except Exception as e:
#             print(f"  [warn] load failed: {e}")
#             out_rows.append({**m, "mean_token_kl": ""})
#             continue
#         tot_kl, tot_tok = 0.0, 0
#         for bi, (b_ids, b_attn) in enumerate(batches()):
#             lg = logits_for(drift, b_ids.to(args.device), b_attn.to(args.device))
#             logp_d = F.log_softmax(lg.float(), dim=-1).cpu()
#             logp_r = ref_logp[bi]
#             p_d = logp_d.exp()
#             kl = (p_d * (logp_d - logp_r)).sum(-1)          # [B,T] KL(d||r)
#             mask = b_attn.bool()
#             tot_kl += float(kl[mask].sum())
#             tot_tok += int(mask.sum())
#         del drift
#         torch.cuda.empty_cache()
#         mean_kl = tot_kl / max(tot_tok, 1)
#         out_rows.append({**m, "mean_token_kl": round(mean_kl, 6)})
#         print(f"  mean_token_kl(d||ref) = {mean_kl:.5f}")

#     fields = list(man[0].keys()) + ["mean_token_kl"]
#     with open(args.out, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=fields)
#         w.writeheader()
#         w.writerows(out_rows)
#     print(f"\nWrote -> {args.out}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
"""
measure_output_kl.py — output-space drift distance (optional, more meaningful)
==============================================================================
Mean per-token KL( p_drifted || p_ref ) between each drifted checkpoint and the
reference, on a fixed probe set — a meaningful x-axis for acceptance-vs-drift.

Robustness fix (Tülu / cross-checkpoint vocab mismatch)
-------------------------------------------------------
The probe is tokenized ONCE with the reference tokenizer, then the SAME ids are
run through every checkpoint (incl. the base model). If the reference tokenizer
carries extra special tokens (Tülu's chat/pad tokens), it can emit an id that
overflows the base model's smaller embedding table -> CUDA device-side assert.
We therefore compute the COMMON vocab V across ref + all checkpoints (from their
configs), force an in-range pad token, clamp any out-of-range id to eos, and
slice every model's logits to [:, :, :V] before softmax, so KL is computed over
the shared vocabulary all checkpoints agree on.

Tip: CUDA_LAUNCH_BLOCKING=1 the first run if asserts persist.

Example:
  python measure_output_kl.py --ref_model allenai/Llama-3.1-Tulu-3-8B \
      --manifest results/drifted_tulu/drift_manifest.csv \
      --probe_jsonl data/benchmarks/probe_mixed.jsonl \
      --n_probe 98 --max_len 256 --out results/drifted_tulu/drift_kl.csv
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _prompt(row):
    for k in ("prompt", "input", "question", "problem_statement", "text", "content"):
        if row.get(k):
            return row[k]
    return ""


def _vocab_of(path):
    try:
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        return int(getattr(cfg, "vocab_size", 0)) or None
    except Exception as e:
        print(f"  [warn] could not read config vocab for {path}: {e}")
        return None


@torch.no_grad()
def logprobs_sliced(model, input_ids, attn, vocab):
    lg = model(input_ids=input_ids, attention_mask=attn).logits
    lg = lg[:, :, :vocab].float()
    return F.log_softmax(lg, dim=-1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--probe_jsonl", required=True)
    ap.add_argument("--n_probe", type=int, default=98)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    man = list(csv.DictReader(open(args.manifest)))

    vocabs = [_vocab_of(args.ref_model)] + [_vocab_of(m["path"]) for m in man]
    vocabs = [v for v in vocabs if v]
    if not vocabs:
        raise SystemExit("Could not determine any vocab sizes from configs.")
    V = min(vocabs)
    print(f"[vocab] common comparison vocab V = {V} (min of {sorted(set(vocabs))})")

    tok = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    if eos_id >= V:
        eos_id = 0
    tok.pad_token_id = eos_id

    rows = [json.loads(l) for l in open(args.probe_jsonl)][: args.n_probe]
    texts = [_prompt(r) for r in rows if _prompt(r)]
    enc = tok(texts, return_tensors="pt", padding="max_length",
              truncation=True, max_length=args.max_len)
    ids, attn = enc["input_ids"], enc["attention_mask"]

    n_oor = int((ids >= V).sum())
    if n_oor:
        print(f"[clamp] {n_oor} token id(s) >= V re-mapped to eos "
              f"(tokenizer/vocab mismatch — expected for Tülu pad/chat tokens)")
        ids = ids.masked_fill(ids >= V, eos_id)

    def batches():
        for i in range(0, len(ids), args.batch_size):
            yield ids[i:i + args.batch_size], attn[i:i + args.batch_size]

    print(f"[ref] {args.ref_model}")
    ref = AutoModelForCausalLM.from_pretrained(
        args.ref_model, torch_dtype=dtype, trust_remote_code=True).to(args.device).eval()
    ref_logp = [logprobs_sliced(ref, b.to(args.device), a.to(args.device), V)
                for b, a in batches()]
    del ref
    torch.cuda.empty_cache()

    out_rows = []
    for m in man:
        print(f"[kl ] {m['tag']}  ({m['path']})")
        try:
            drift = AutoModelForCausalLM.from_pretrained(
                m["path"], torch_dtype=dtype, trust_remote_code=True).to(args.device).eval()
        except Exception as e:
            print(f"  [warn] load failed: {e}")
            out_rows.append({**m, "mean_token_kl": ""})
            continue
        tot_kl, tot_tok = 0.0, 0
        for bi, (b, a) in enumerate(batches()):
            logp_d = logprobs_sliced(drift, b.to(args.device), a.to(args.device), V)
            logp_r = ref_logp[bi]
            p_d = logp_d.exp()
            kl = (p_d * (logp_d - logp_r)).sum(-1)
            mask = a.bool()
            tot_kl += float(kl[mask].sum())
            tot_tok += int(mask.sum())
        del drift
        torch.cuda.empty_cache()
        mean_kl = tot_kl / max(tot_tok, 1)
        out_rows.append({**m, "mean_token_kl": round(mean_kl, 6)})
        print(f"  mean_token_kl(d||ref) = {mean_kl:.5f}")

    fields = list(man[0].keys()) + ["mean_token_kl"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
