"""Lightweight token-regime features for controllers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math


def entropy_from_probs(probs: Sequence[float], *, base: float = math.e) -> float:
    """Return entropy of a probability vector."""

    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("probability vector must have positive mass")
    log_base = math.log(base)
    entropy = 0.0
    for prob in probs:
        if prob < 0.0:
            raise ValueError("probabilities must be non-negative")
        if prob == 0.0:
            continue
        normalized = prob / total
        entropy -= normalized * (math.log(normalized) / log_base)
    return entropy


def repetition_density(tokens: Sequence[int | str], *, window: int = 128, lookback: int = 128) -> float:
    """Fraction of recent tokens that occurred earlier in a lookback window."""

    if window <= 0 or lookback <= 0:
        raise ValueError("window and lookback must be positive")
    if not tokens:
        return 0.0

    recent = list(tokens[-window:])
    prefix_end = max(0, len(tokens) - len(recent))
    history = Counter(tokens[max(0, prefix_end - lookback) : prefix_end])
    if not recent:
        return 0.0
    repeated = 0
    for token in recent:
        if history[token] > 0:
            repeated += 1
        history[token] += 1
    return repeated / len(recent)
