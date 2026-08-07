# APEX evaluation suite

This folder is a paper-oriented evaluation harness for APEX. It does not replace the runtime code under `src/apexp`; it organizes training curves, hot-path learned-controller runs, fixed-baseline summaries, candidate-k ablations, temperature ablations, and optional official external baselines.

## What this suite produces

1. **Training study** for recent survival-utility controller runs:
   - training curves for `speed`, `balanced`, and `efficient` variants
   - best checkpoint summary across v4/v4b roots
   - model-selection CSVs for offline ranking metrics

2. **APEX learned hot-path study**:
   - all workloads
   - temperature sweep: `0.0`, `0.1`, `0.5`
   - candidate-k ablation: `{8,16}`, `{4,8,16}`, `{1..16}`
   - model variants: current v4 speed/balanced/efficient and v4b variants when present

3. **Fixed-baseline study**:
   - AR / slow-only where available
   - fixed n-gram speculative decoding
   - fixed draft-model speculative decoding
   - fixed EAGLE3 speculative decoding
   - fixed-k summaries by method, k, workload, and temperature when the corresponding result folders exist

4. **External official-baseline check**:
   - BanditSpec is treated as the only mandatory closely related official adaptive-controller baseline.
   - The harness does not vendor the BanditSpec repo. Set `BANDITSPEC_ROOT=/path/to/BanditSpec` and use `05_check_official_baselines.py` to verify availability.
   - DSDE/AdaEAGLE/HeteroSpec-style methods are tracked as related work unless an official runnable implementation is supplied.

## Main commands

Run from repository root:

```bash
cd /fsx/jmanvi/Internship_project/ASD
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
```

### 1. Training graphs and offline model-selection table

```bash
python3 evaluation/01_plot_training.py \
  --roots \
    results/apexp_block/config_large_500/week2/survival_utility_controller_v4 \
    results/apexp_block/config_large_500/week2/survival_utility_controller_v4b_rankgroup \
  --out-dir results/apexp_block/config_large_500/evaluation/training_study
```

Outputs:

```text
results/apexp_block/config_large_500/evaluation/training_study/all_training_history.csv
results/apexp_block/config_large_500/evaluation/training_study/best_model_summary.csv
results/apexp_block/config_large_500/evaluation/training_study/plots/*.png
```

### 2. Generate Slurm jobs for learned true-block APEX hot runs

```bash
python3 evaluation/02_make_hot_eval_jobs.py \
  --out-dir slurm_logs/evaluation_jobs \
  --result-root results/apexp_block/config_large_500/evaluation/hot_runs \
  --slow-router-dir results/apexp_block/config_large_500/week2/request_prompt_router_d128 \
  --model-dirs \
    results/apexp_block/config_large_500/week2/survival_utility_controller_v4/speed_first \
    results/apexp_block/config_large_500/week2/survival_utility_controller_v4/balanced \
    results/apexp_block/config_large_500/week2/survival_utility_controller_v4/efficient \
    results/apexp_block/config_large_500/week2/survival_utility_controller_v4b_rankgroup/speed_lr5e4_rank2 \
  --workload-files \
    data/benchmarks/By_split_phase_1/code_gen_test.jsonl \
    data/benchmarks/By_split_phase_1/mathematical_reasoning_test.jsonl \
    data/benchmarks/By_split_phase_1/long_context_completion_test.jsonl \
    data/benchmarks/By_split_phase_1/long_chain_reasoning_test.jsonl \
    data/benchmarks/By_split_phase_1/long_horizon_swe_test.jsonl \
    data/benchmarks/By_split_phase_1/hardware_gen_test.jsonl \
    data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl \
    data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl \
  --temps 0.0 0.1 0.5 \
  --candidate-sets k8_16=8,16 k4_8_16=4,8,16 k1_16=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --max-prompts-per-file 0
```

Submit generated jobs:

```bash
for f in slurm_logs/evaluation_jobs/*.sbatch; do sbatch "$f"; done
```

For a smoke run, set `--max-prompts-per-file 20` first.

### 3. Summarize all hot runs

```bash
python3 evaluation/03_summarize_hot_runs.py \
  --runs-root results/apexp_block/config_large_500/evaluation/hot_runs \
  --out-dir results/apexp_block/config_large_500/evaluation/hot_summary
```

Outputs:

```text
request_overall_all_runs.csv
block_overall_all_runs.csv
request_by_workload_all_runs.csv
block_by_workload_all_runs.csv
candidate_k_ablation.csv
temperature_ablation.csv
plots/*.png
```

### 4. Summarize fixed baselines

Point this to any fixed-baseline roots that already exist:

```bash
python3 evaluation/04_collect_fixed_baselines.py \
  --roots results/apexp_block/config_large_500 \
  --out-dir results/apexp_block/config_large_500/evaluation/fixed_baselines
```

The script is intentionally tolerant: it recursively searches for request/block summaries and result JSON files.

### 5. Check official external baselines

```bash
export BANDITSPEC_ROOT=/path/to/BanditSpec
python3 evaluation/05_check_official_baselines.py \
  --out-dir results/apexp_block/config_large_500/evaluation/external_baselines
```

This only verifies official-code availability and writes an implementation-readiness report. It does not silently replace missing official baselines with unofficial approximations.

## Metrics

The suite reports:

- `mean_tps`, `median_tps`
- `speedup_vs_slow_only_tps`
- `latency_speedup_vs_slow_only`
- `wasted_token_rate`
- `wasted_token_pct`
- `accepted_per_block`
- `full_accept_rate`
- `mean_k`
- `STE_vs_slow_only`
- `ATE_vs_slow_only`

For paper tables, prioritize:

```text
raw TPS / speedup
wasted token %
accepted/block
STE
ATE
chosen-k distribution
policy runtime overhead
```

## Recommended paper tables

1. Main comparison: fixed baselines vs APEX learned true-block.
2. Controller objective study: speed vs balanced vs efficient.
3. Candidate-k ablation: `{8,16}` vs `{4,8,16}` vs `{1..16}`.
4. Temperature ablation: `0.0`, `0.1`, `0.5`.
5. Workload breakdown.
6. Official external-baseline readiness / BanditSpec comparison when runnable.
