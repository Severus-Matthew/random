"""Empirical hazard estimation from speculative decoding traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from apex.survival import expected_accepted_length, survival_curve
from apex.traces import BlockTrace


@dataclass(frozen=True)
class HazardEstimate:
    """Position-wise conditional acceptance and rejection estimates."""

    accept_probs: tuple[float, ...]
    hazards: tuple[float, ...]
    survival: tuple[float, ...]
    exposures: tuple[int, ...]
    accepts: tuple[float, ...]

    @property
    def expected_len(self) -> float:
        return expected_accepted_length(self.accept_probs)


class BetaHazardEstimator:
    """Beta-Bernoulli estimator for p_j with censored trace handling.

    At position j, a block contributes an exposure only if verification reached
    position j. The observed label is:
    - 1 if first_rejection > j, including full acceptance;
    - 0 if first_rejection == j;
    - unobserved if first_rejection < j.
    """

    def __init__(self, max_depth: int, *, alpha: float = 1.0, beta: float = 1.0) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be positive")
        self.max_depth = max_depth
        self.alpha = alpha
        self.beta = beta
        self._accepts = [0.0] * max_depth
        self._exposures = [0] * max_depth

    def update(self, trace: BlockTrace) -> None:
        for position in range(min(trace.draft_len, self.max_depth)):
            label = trace.observed_label(position)
            if label is None:
                continue
            self._exposures[position] += 1
            self._accepts[position] += label

    def update_many(self, traces: Iterable[BlockTrace]) -> None:
        for trace in traces:
            self.update(trace)

    def estimate(self, depth: int | None = None) -> HazardEstimate:
        if depth is None:
            depth = self.max_depth
        if not 0 <= depth <= self.max_depth:
            raise ValueError("depth must lie in [0, max_depth]")

        accept_probs: list[float] = []
        for accepts, exposures in zip(self._accepts[:depth], self._exposures[:depth], strict=True):
            posterior_mean = (self.alpha + accepts) / (self.alpha + self.beta + exposures)
            accept_probs.append(posterior_mean)

        hazards = [1.0 - prob for prob in accept_probs]
        survival = survival_curve(accept_probs)
        return HazardEstimate(
            accept_probs=tuple(accept_probs),
            hazards=tuple(hazards),
            survival=tuple(survival),
            exposures=tuple(self._exposures[:depth]),
            accepts=tuple(self._accepts[:depth]),
        )
