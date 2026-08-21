# APEX Speculative Decoding Core

This repository contains the first implementation layer for APEX: acceptance-survival math, trace handling, online hazard estimation, and depth controllers.

The implementation follows the corrected zero-based convention from the proposal:

- `p[0]` is the acceptance probability of the first drafted token.
- `S[j] = prod_{i=0}^j p[i]` is the probability that the draft survives through position `j`.
- `S[-1] = 1` is only the empty-prefix boundary value.
- accepted length `L` lies in `{0, ..., k}`.

The corrected expected accepted length is:

```text
E[L | k] =
sum_{j=0}^{k-1} j * prod_{i=0}^{j-1} p_i * (1 - p_j)
+ k * prod_{i=0}^{k-1} p_i
= sum_{j=0}^{k-1} prod_{i=0}^j p_i
```

## Basic commands

Choose a depth from a predicted acceptance curve:

```bash
python -m apex.cli choose-depth --accept-probs 0.8,0.7,0.5,0.2 --depths 1,2,4
```

Fit empirical hazards from JSONL traces:

```bash
python -m apex.cli fit-hazard --traces traces.jsonl --max-depth 16 --output hazard.json
```

Summarize trace metrics:

```bash
python -m apex.cli summarize --traces examples/sample_traces.jsonl --max-depth 4
```

Replay online depth choices from historical traces:

```bash
python -m apex.cli replay --traces examples/sample_traces.jsonl --max-depth 4 --depths 1,2,4
```

## Training code

The repository now has two training tracks:

- `apex.training`: acceptance-utility drafter training, adapter/LoRA conditioning, and neural hazard-model loss.
- `apex.rl`: controller training through contextual bandits first, then policy-gradient/PPO scaffolding.

The LoRA and neural RL modules import `torch`, `transformers`, and `peft` lazily. They are safe to import in a CPU-only environment, but actual training requires installing those packages.

Precompute drafter weights from examples:

```python
from apex.training.drafter_trainer import build_weighted_labels

rows = build_weighted_labels("examples/sample_training.jsonl", rejection_bonus=1.0)
```

Or from the CLI:

```bash
python -m apex.cli prepare-drafter-data \
  --examples examples/sample_training.jsonl \
  --output output/weighted_training.jsonl
```

Use a contextual bandit over depths:

```python
from apex.rl.bandit import LinUCBBandit, make_trace_context
from apex.rl.rewards import block_reward
from apex.traces import read_jsonl

traces = read_jsonl("examples/sample_traces.jsonl")
context = make_trace_context(traces[0])
bandit = LinUCBBandit([1, 2, 4, 8, 16], context_dim=len(context))
bandit.update(context, traces[0].draft_len, block_reward(traces[0]))
```

Or train and save the bandit state:

```bash
python -m apex.cli train-bandit \
  --traces examples/sample_traces.jsonl \
  --depths 1,2,4 \
  --output output/bandit.json
```

## Trace schema

Each speculative block is represented as a JSON object compatible with `apex.traces.BlockTrace`.

```json
{
  "request_id": "req-0",
  "block_id": 0,
  "method": "ngram_sd",
  "draft_len": 4,
  "first_rejection": 2,
  "latency_ms": 12.5,
  "entropy": 1.4,
  "repetition_density": 0.42,
  "temperature": 0.0,
  "regime": "copy_like",
  "metadata": {}
}
```

Here `first_rejection=2` means positions `0` and `1` were accepted, position `2` was rejected, and position `3` was unobserved.
