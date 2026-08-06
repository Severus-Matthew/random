#!/usr/bin/env python
"""
make_drifted_models.py — materialize "policy-drift" target checkpoints
======================================================================
The staleness hypothesis behind Phase 3 is: as the target (policy) model
drifts during RL training, a fixed drafter's acceptance rate decays. We cannot
run RL yet, so we *proxy* policy drift on the inference-only setup by producing
controlled perturbations of the target model and measuring how SD acceptance
degrades for a fixed drafter.

Three drift modes (all keep the SAME drafter; only the TARGET changes):

  interp   theta(a) = (1-a)*theta_base + a*theta_ref
           Interpolate between a far checkpoint (base) and the reference the
           drafter is aligned to (ref, e.g. the instruct model the EAGLE3 head
           was trained on). a=1.0 -> fresh (no drift); a=0.0 -> max drift.
           This is the most realistic proxy: a real "training trajectory"
           between two genuine checkpoints of the same architecture.

  noise    theta' = theta_ref + sigma * std(theta_ref) * eps,  eps ~ N(0, I)
           Synthetic, fully controlled isotropic drift. sigma=0 -> fresh.
           Perturbs weight matrices (ndim>=2) only by default.

  external Register an already-existing checkpoint path as a drift point
           (e.g. a real RLHF/SFT variant) and measure its distance from ref.

For every materialized checkpoint we record the *relative weight-space
distance* from the reference (cheap, exact). Output-space KL is measured
separately by measure_output_kl.py (optional, more expensive, more meaningful).

Outputs:
  <out_root>/<tag>/                      a HF checkpoint vLLM can load
  <out_root>/drift_manifest.csv          tag, mode, param, rel_weight_dist, path

Example:
  python make_drifted_models.py \
      --ref_model Qwen/Qwen3-8B --base_model Qwen/Qwen3-8B-Base \
      --mode interp --alphas 1.0 0.9 0.75 0.5 0.25 0.1 0.0 \
      --out_root /scratch/drifted --dtype bfloat16
  python make_drifted_models.py --ref_model Qwen/Qwen3-8B \
      --mode noise --sigmas 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 \
      --out_root /scratch/drifted

RAM note: interp loads two full models on CPU (~2x model size). For an 8B in
bf16 that is ~32 GB host RAM; ensure the node has it. Materialize + run + delete
one level at a time via run_staleness_study.py --cleanup if disk is tight
(each 8B bf16 checkpoint is ~16 GB).
"""
import argparse
import csv
import math
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _rel_distance(numer_sq: float, denom_sq: float) -> float:
    return math.sqrt(numer_sq) / math.sqrt(denom_sq) if denom_sq > 0 else 0.0


def _perturbable(name: str, param: torch.Tensor, skip_norms: bool) -> bool:
    if param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if param.ndim < 2:                 # skip biases / 1-D vectors
        return False
    if skip_norms and ("norm" in name.lower() or "ln" in name.lower()):
        return False
    return True


def build_interp(ref_model, base_model, alpha, skip_norms):
    """In place: set ref params to (1-a)*base + a*ref. Returns rel distance from ref."""
    ref_sd = dict(ref_model.named_parameters())
    base_sd = dict(base_model.named_parameters())
    numer_sq, denom_sq = 0.0, 0.0
    missing = [k for k in ref_sd if k not in base_sd]
    if missing:
        raise SystemExit(f"[interp] {len(missing)} params not in base "
                         f"(e.g. {missing[:3]}). Architectures must match.")
    with torch.no_grad():
        for name, p_ref in ref_sd.items():
            if not _perturbable(name, p_ref, skip_norms):
                continue
            p_base = base_sd[name]
            r = p_ref.detach().float()
            b = p_base.detach().float()
            new = (1.0 - alpha) * b + alpha * r
            numer_sq += float(((new - r) ** 2).sum())
            denom_sq += float((r ** 2).sum())
            p_ref.data.copy_(new.to(p_ref.dtype))
    return _rel_distance(numer_sq, denom_sq)


def build_noise(ref_model, sigma, skip_norms, seed):
    g = torch.Generator().manual_seed(seed)
    numer_sq, denom_sq = 0.0, 0.0
    with torch.no_grad():
        for name, p in ref_model.named_parameters():
            if not _perturbable(name, p, skip_norms):
                continue
            r = p.detach().float()
            std = float(r.std())
            if std == 0 or sigma == 0:
                denom_sq += float((r ** 2).sum())
                continue
            eps = torch.randn(r.shape, generator=g)
            delta = sigma * std * eps
            numer_sq += float((delta ** 2).sum())
            denom_sq += float((r ** 2).sum())
            p.data.copy_((r + delta).to(p.dtype))
    return _rel_distance(numer_sq, denom_sq)


def save_checkpoint(model, tokenizer, out_dir: Path):
    """Save a vLLM/HF-loadable checkpoint WITHOUT calling model.save_pretrained.

    save_pretrained() routes through accelerate's extract_model_from_parallel,
    which imports deepspeed; on nodes that have the CUDA runtime but not nvcc,
    deepspeed's import-time compatibility check crashes. We sidestep that
    entirely: config + generation_config + tokenizer are plain JSON (no
    deepspeed), and weights are written directly with safetensors.
    """
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) config / generation config (pure JSON; no deepspeed import)
    model.config.save_pretrained(out_dir)
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            gen_cfg.save_pretrained(out_dir)
        except Exception:
            pass

    # 2) weights -> single model.safetensors, with tied-weight de-dup
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    sd = model.state_dict()
    seen, to_save = {}, {}
    for name, t in sd.items():
        # HF omits tied lm_head when embeddings are tied; mirror that
        if tied and name in ("lm_head.weight",):
            continue
        t = t.detach().cpu().contiguous()
        ptr = t.data_ptr()
        if ptr in seen:               # any other accidental storage sharing
            t = t.clone()
        seen[ptr] = name
        to_save[name] = t
    save_file(to_save, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    # 3) tokenizer (tokenizers package; no deepspeed)
    tokenizer.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_model", required=True,
                    help="Model the drafter is aligned to (fresh / no-drift endpoint).")
    ap.add_argument("--base_model", default=None,
                    help="Far endpoint for interpolation (e.g. *-Base).")
    ap.add_argument("--mode", choices=["interp", "noise", "external"], required=True)
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0])
    ap.add_argument("--sigmas", type=float, nargs="*",
                    default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--external_paths", nargs="*", default=[],
                    help="For mode=external: existing checkpoint dirs to register.")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--skip_norms", action="store_true", default=True)
    ap.add_argument("--no_skip_norms", dest="skip_norms", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family", default="",
                    help="Optional family label; prefixes tags and is written as "
                         "a 'family' column so multiple drift families don't collide.")
    args = ap.parse_args()

    fam_prefix = f"{args.family}__" if args.family else ""

    dtype = getattr(torch, args.dtype)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "drift_manifest.csv"
    rows = []

    tok = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)

    def _load(name_or_path):
        return AutoModelForCausalLM.from_pretrained(
            name_or_path, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True)

    if args.mode == "interp":
        if not args.base_model:
            raise SystemExit("--base_model required for mode=interp")
        print(f"[load] base = {args.base_model}")
        base = _load(args.base_model)
        for a in args.alphas:
            print(f"[load] ref  = {args.ref_model}  (alpha={a})")
            ref = _load(args.ref_model)                  # reload clean ref each time
            dist = build_interp(ref, base, a, args.skip_norms)
            tag = f"{fam_prefix}interp_a{a:.2f}"
            path = out_root / tag
            save_checkpoint(ref, tok, path)
            del ref
            rows.append(dict(tag=tag, family=args.family, mode="interp", param=a,
                             rel_weight_dist=round(dist, 6), path=str(path)))
            print(f"  -> {tag}  rel_weight_dist_from_ref={dist:.5f}")
        del base

    elif args.mode == "noise":
        for s in args.sigmas:
            print(f"[load] ref  = {args.ref_model}  (sigma={s})")
            ref = _load(args.ref_model)
            dist = build_noise(ref, s, args.skip_norms, args.seed)
            tag = f"{fam_prefix}noise_s{s:g}"
            path = out_root / tag
            save_checkpoint(ref, tok, path)
            del ref
            rows.append(dict(tag=tag, family=args.family, mode="noise", param=s,
                             rel_weight_dist=round(dist, 6), path=str(path)))
            print(f"  -> {tag}  rel_weight_dist_from_ref={dist:.5f}")

    elif args.mode == "external":
        ref = _load(args.ref_model)
        ref_params = {n: p.detach().float() for n, p in ref.named_parameters()}
        del ref
        for ext in args.external_paths:
            tag = f"{fam_prefix}external_" + Path(ext).name
            try:
                m = _load(ext)
                numer_sq = denom_sq = 0.0
                for n, p in m.named_parameters():
                    if n in ref_params and _perturbable(n, p, args.skip_norms):
                        r = ref_params[n]
                        numer_sq += float(((p.detach().float() - r) ** 2).sum())
                        denom_sq += float((r ** 2).sum())
                dist = _rel_distance(numer_sq, denom_sq)
                del m
            except Exception as e:
                print(f"  [warn] could not measure {ext}: {e}")
                dist = float("nan")
            rows.append(dict(tag=tag, family=args.family, mode="external", param="",
                             rel_weight_dist=round(dist, 6), path=str(ext)))
            print(f"  -> {tag}  rel_weight_dist_from_ref={dist:.5f}")

    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "family", "mode", "param",
                                          "rel_weight_dist", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote manifest -> {manifest}  ({len(rows)} drift points)")


if __name__ == "__main__":
    main()
