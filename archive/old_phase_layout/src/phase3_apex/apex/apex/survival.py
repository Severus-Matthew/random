"""Survival math for speculative decoding.

Conventions:
- Draft positions are zero-based: j in {0, ..., k - 1}.
- p[0] is the first-token conditional acceptance probability.
- S[j] is the probability that positions 0..j all survive.
- S[-1] is represented only by the empty product value 1.0.
- L is the accepted draft length and lies in {0, ..., k}.
"""

from __future__ import annotations

from collections.abc import Sequence


def _check_probabilities(values: Sequence[float], name: str) -> None:
    for idx, value in enumerate(values):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{idx}]={value!r} is outside [0, 1]")


def survival_curve(accept_probs: Sequence[float]) -> list[float]:
    """Return S_j = prod_{i=0}^j p_i for j=0..k-1."""

    _check_probabilities(accept_probs, "accept_probs")
    curve: list[float] = []
    running = 1.0
    for prob in accept_probs:
        running *= prob
        curve.append(running)
    return curve


def accepted_length_pmf(accept_probs: Sequence[float]) -> list[float]:
    """Return P(L=j) for j=0..k under the corrected zero-based formula.

    For j < k:
        P(L=j) = prod_{i=0}^{j-1} p_i * (1 - p_j)
    For j = k:
        P(L=k) = prod_{i=0}^{k-1} p_i
    """

    _check_probabilities(accept_probs, "accept_probs")
    pmf: list[float] = []
    survived_prefix = 1.0
    for prob in accept_probs:
        pmf.append(survived_prefix * (1.0 - prob))
        survived_prefix *= prob
    pmf.append(survived_prefix)
    return pmf


def expected_accepted_length_exact(accept_probs: Sequence[float]) -> float:
    """Compute E[L] from the exact accepted-length distribution."""

    return sum(length * mass for length, mass in enumerate(accepted_length_pmf(accept_probs)))


def expected_accepted_length(accept_probs: Sequence[float]) -> float:
    """Compute E[L] using the equivalent survival-sum form.

    E[L | k] = sum_{j=0}^{k-1} S_j = sum_{j=0}^{k-1} prod_{i=0}^j p_i.
    """

    return sum(survival_curve(accept_probs))


def expected_accepted_length_from_hazards(hazards: Sequence[float]) -> float:
    """Compute E[L] from rejection hazards h_j = 1 - p_j."""

    _check_probabilities(hazards, "hazards")
    return expected_accepted_length([1.0 - hazard for hazard in hazards])


def marginal_depth_value(accept_probs: Sequence[float], next_accept_prob: float | None = None) -> float:
    """Return the expected accepted-length gain from adding the next position.

    If accept_probs already contains p_0..p_k, this returns S_k, the gain from
    increasing depth from k to k+1. If accept_probs contains only p_0..p_{k-1},
    pass next_accept_prob=p_k.
    """

    if next_accept_prob is not None:
        _check_probabilities([next_accept_prob], "next_accept_prob")
        values = [*accept_probs, next_accept_prob]
    else:
        values = list(accept_probs)
    curve = survival_curve(values)
    return curve[-1] if curve else 0.0


def normalize_pmf(pmf: Sequence[float], *, atol: float = 1e-9) -> None:
    """Validate that a PMF is normalized within numerical tolerance."""

    total = sum(pmf)
    if abs(total - 1.0) > atol:
        raise ValueError(f"pmf sums to {total}, expected 1.0")
