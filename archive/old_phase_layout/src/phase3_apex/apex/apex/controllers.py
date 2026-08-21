"""Depth controllers for APEX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from apex.survival import expected_accepted_length, marginal_depth_value


@dataclass(frozen=True)
class DepthDecision:
    depth: int
    expected_len: float
    utility: float
    accept_probs: tuple[float, ...]


@dataclass(frozen=True)
class CostModel:
    """Simple speculative-block cost model."""

    verifier_ms: float = 1.0
    draft_ms_per_token: float = 0.0
    controller_ms: float = 0.0
    fixed_overhead_ms: float = 0.0

    def cost(self, depth: int) -> float:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        value = self.verifier_ms + self.fixed_overhead_ms + self.controller_ms
        value += depth * self.draft_ms_per_token
        if value <= 0.0:
            raise ValueError("cost must be positive")
        return value


class SurvivalDepthController:
    """Choose k by maximizing expected accepted length per cost."""

    def __init__(
        self,
        candidate_depths: Sequence[int],
        *,
        cost_model: CostModel | None = None,
        min_depth: int = 1,
    ) -> None:
        if not candidate_depths:
            raise ValueError("candidate_depths cannot be empty")
        unique = sorted(set(candidate_depths))
        if unique[0] < 0:
            raise ValueError("candidate_depths must be non-negative")
        if min_depth < 0:
            raise ValueError("min_depth must be non-negative")
        self.candidate_depths = tuple(unique)
        self.cost_model = cost_model or CostModel()
        self.min_depth = min_depth

    def choose(self, accept_probs: Sequence[float]) -> DepthDecision:
        if not accept_probs:
            return DepthDecision(depth=0, expected_len=0.0, utility=0.0, accept_probs=())

        best: DepthDecision | None = None
        for depth in self.candidate_depths:
            if depth < self.min_depth:
                continue
            truncated = tuple(accept_probs[:depth])
            if len(truncated) < depth:
                continue
            expected_len = expected_accepted_length(truncated)
            utility = expected_len / self.cost_model.cost(depth)
            decision = DepthDecision(depth=depth, expected_len=expected_len, utility=utility, accept_probs=truncated)
            if best is None or (decision.utility, decision.expected_len, -decision.depth) > (
                best.utility,
                best.expected_len,
                -best.depth,
            ):
                best = decision

        if best is None:
            return DepthDecision(depth=0, expected_len=0.0, utility=0.0, accept_probs=())
        return best


class ThresholdDepthController:
    """Stop adding positions when survival gain falls below marginal cost threshold."""

    def __init__(self, *, min_depth: int = 1, max_depth: int = 16, min_marginal_value: float = 0.05) -> None:
        if min_depth < 0 or max_depth < min_depth:
            raise ValueError("require 0 <= min_depth <= max_depth")
        if not 0.0 <= min_marginal_value <= 1.0:
            raise ValueError("min_marginal_value must lie in [0, 1]")
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_marginal_value = min_marginal_value

    def choose(self, accept_probs: Sequence[float]) -> DepthDecision:
        depth = min(self.min_depth, len(accept_probs), self.max_depth)
        for candidate in range(depth, min(len(accept_probs), self.max_depth)):
            gain = marginal_depth_value(accept_probs[: candidate + 1])
            if candidate + 1 <= self.min_depth or gain >= self.min_marginal_value:
                depth = candidate + 1
            else:
                break

        truncated = tuple(accept_probs[:depth])
        expected_len = expected_accepted_length(truncated)
        return DepthDecision(depth=depth, expected_len=expected_len, utility=expected_len, accept_probs=truncated)


def choose_regime(features: Mapping[str, float], *, low_entropy: float = 1.0, high_entropy: float = 3.0, high_repetition: float = 0.35) -> str:
    """Rule-based slow controller for the regime code r_t."""

    entropy = features.get("entropy", 0.0)
    repetition = features.get("repetition_density", 0.0)
    if entropy < low_entropy and repetition >= high_repetition:
        return "copy_like"
    if entropy >= high_entropy:
        return "conservative"
    return "neural"
