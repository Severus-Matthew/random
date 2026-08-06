# Staleness Proxy Study (`src/phase2/staleness_proxy/`)

**Question (the Phase-3 go/no-go):** does a fixed drafter's acceptance rate decay
as the target/policy model drifts? The "online speculative distillation" thesis
needs YES. We can't run RL yet, so we proxy policy drift on the inference-only
setup and measure the decay curve. This now runs across **three drift families**
so the result rests on both controlled axes and a *real* RLHF trajectory.

## Drift families

| Family | Axis | Drafters | EAGLE-3 head |
|---|---|---|---|
| `qwen_interp` | Base ↔ Qwen3-8B interpolation | ngram, draft_sd, eagle3 | `RedHatAI/Qwen3-8B-speculator.eagle3` |
| `llama_interp` | Llama-3.1-8B Base ↔ Instruct interpolation | ngram, draft_sd, eagle3 | `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` |
| `tulu_trajectory` | **REAL** path: Base→SFT→DPO→RLVR (Tülu 3) | ngram, draft_sd | — (head not aligned to Tülu) |

The two interp families are controlled/dense; the Tülu family is a genuine
post-training trajectory (4 real checkpoints, no training on your side) and is
the convincing rebuttal to NeMo-RL's "staleness is small" framing. eagle3 is
auto-skipped on Tülu (no aligned head); n-gram (model-free) and draft_sd (small
Llama draft) run there cleanly.

**Expected headline:** n-gram ~flat (rebuilt from context, target-independent),
eagle3/draft_sd decay — the *slope* is the staleness sensitivity. If even eagle3
is flat across both a synthetic axis and the real Tülu path, rethink the thesis.

## Build the drift checkpoints (one per family)

```bash
ROOT=/fsx/jmanvi/Internship_project/ASD

# Qwen interp (you already have the Qwen eagle3 head)
python make_drifted_models.py --family qwen_interp \
    --ref_model Qwen/Qwen3-8B --base_model Qwen/Qwen3-8B-Base --mode interp \
    --alphas 1.0 0.9 0.75 0.5 0.25 0.1 0.0 --out_root $ROOT/results/drifted_qwen

# Llama interp
python make_drifted_models.py --family llama_interp \
    --ref_model meta-llama/Llama-3.1-8B-Instruct --base_model meta-llama/Llama-3.1-8B \
    --mode interp --alphas 1.0 0.9 0.75 0.5 0.25 0.1 0.0 \
    --out_root $ROOT/results/drifted_llama

# Tülu REAL trajectory (ref = final RLVR; externals = staler points)
python make_drifted_models.py --family tulu_trajectory \
    --ref_model allenai/Llama-3.1-Tulu-3-8B --mode external \
    --external_paths meta-llama/Llama-3.1-8B \
        allenai/Llama-3.1-Tulu-3-8B-SFT allenai/Llama-3.1-Tulu-3-8B-DPO \
    --out_root $ROOT/results/drifted_tulu
```

(Optional) output-KL drift distance per family:
```bash
# 1. build ONE balanced cross-workload probe (14 prompts x 7 workloads ≈ 98)
python build_mixed_probe.py --data_dir $ROOT/data/benchmarks/By_split_phase_1 \
    --per_workload 14 --out $ROOT/data/benchmarks/probe_mixed.jsonl

# 2. one KL run per family against that mixed probe
python measure_output_kl.py --ref_model Qwen/Qwen3-8B \
    --manifest $ROOT/results/drifted_qwen/drift_manifest.csv \
    --probe_jsonl $ROOT/data/benchmarks/probe_mixed.jsonl --n_probe 98 \
    --max_len 256 --out $ROOT/results/drifted_qwen/drift_kl.csv
```
Use a *mixed* probe, not math-only: RLHF drift moves reasoning/format tokens far
more than boilerplate, so a single-domain KL mis-calibrates the acceptance-vs-KL
x-axis on other workloads. KL is optional — `rel_weight_dist` (free at checkpoint
build) already works as an x-axis; KL just makes it output-space-meaningful for
the final figures.

## Run + aggregate

```bash
# plan first (writes results/staleness_index.csv — the authoritative
# experiment->family->drift map the aggregator uses)
python run_staleness_study.py --config staleness_config.json \
    --repo_root $ROOT --gpus 8 --plan_only
# execute across 8 GPUs, deleting each ckpt after its cells finish
python run_staleness_study.py --config staleness_config.json \
    --repo_root $ROOT --gpus 8 --cleanup

# aggregate — joins runs to the index by experiment (no manifest concat needed),
# and optionally folds in per-family output-KL via a glob
python aggregate_staleness.py \
    --runs_dir $ROOT/results/Runs/Phase2 \
    --index $ROOT/results/staleness_index.csv \
    --kl '$ROOT/results/drifted_*/drift_kl.csv' \
    --out_dir $ROOT/results/staleness
```

Outputs: `staleness_slopes.csv` (headline per family×drafter), `staleness_retention.csv`,
`staleness_cells.csv`, `staleness_position_decay.csv`.

## Notes / fixes
- **`save_pretrained` deepspeed/nvcc crash is fixed.** `make_drifted_models.py`
  now writes checkpoints directly via safetensors (config + tokenizer are plain
  JSON), so it never imports deepspeed — avoids the `nvcc not found` failure on
  nodes with the CUDA runtime but no compiler. Tied embeddings are de-duplicated.
- Smoke-test one `eagle3` cell on a drifted checkpoint before the full grid
  (confirm vLLM loads the head against perturbed weights).
- This is a *proxy*: it measures staleness *sensitivity*, validated later by the
  real RL loop. The Tülu family is the closest-to-real axis and the strongest
  evidence.
- Grid sizes (with 7+7+4 drift points): ~1k engine builds. For a fast first
  signal, run `tulu_trajectory` + `S1` only, then expand.
