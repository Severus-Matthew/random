"""Contextual bandit controllers for APEX depth selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from apex.traces import BlockTrace


def make_trace_context(trace: BlockTrace, *, max_depth: int = 16) -> np.ndarray:
    """Feature vector used by the bandit from one block trace."""

    draft_ratio = trace.draft_len / max(1, max_depth)
    return np.asarray(
        [
            1.0,
            trace.entropy if trace.entropy is not None else 0.0,
            trace.repetition_density if trace.repetition_density is not None else 0.0,
            trace.temperature if trace.temperature is not None else 0.0,
            trace.target_drift if trace.target_drift is not None else 0.0,
            draft_ratio,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BanditDecision:
    depth: int
    score: float
    predicted_reward: float
    uncertainty: float


class LinUCBBandit:
    """LinUCB contextual bandit over candidate speculation depths."""

    def __init__(self, candidate_depths: Sequence[int], context_dim: int, *, alpha: float = 1.0, ridge: float = 1.0) -> None:
        if not candidate_depths:
            raise ValueError("candidate_depths cannot be empty")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.candidate_depths = tuple(sorted(set(candidate_depths)))
        self.context_dim = context_dim
        self.alpha = alpha
        self._a = {depth: ridge * np.eye(context_dim, dtype=np.float64) for depth in self.candidate_depths}
        self._b = {depth: np.zeros(context_dim, dtype=np.float64) for depth in self.candidate_depths}

    def choose(self, context: np.ndarray) -> BanditDecision:
        context = self._validate_context(context)
        best: BanditDecision | None = None
        for depth in self.candidate_depths:
            a_inv = np.linalg.inv(self._a[depth])
            theta = a_inv @ self._b[depth]
            predicted = float(theta @ context)
            uncertainty = float(np.sqrt(context @ a_inv @ context))
            score = predicted + self.alpha * uncertainty
            decision = BanditDecision(depth=depth, score=score, predicted_reward=predicted, uncertainty=uncertainty)
            if best is None or decision.score > best.score:
                best = decision
        assert best is not None
        return best

    def update(self, context: np.ndarray, depth: int, reward: float) -> None:
        if depth not in self._a:
            raise ValueError(f"unknown depth {depth}")
        context = self._validate_context(context)
        self._a[depth] += np.outer(context, context)
        self._b[depth] += reward * context

    def fit(self, contexts: Iterable[np.ndarray], depths: Iterable[int], rewards: Iterable[float]) -> None:
        for context, depth, reward in zip(contexts, depths, rewards, strict=True):
            self.update(context, depth, reward)

    def state_dict(self) -> dict[str, object]:
        return {
            "candidate_depths": self.candidate_depths,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "a": {str(depth): value.tolist() for depth, value in self._a.items()},
            "b": {str(depth): value.tolist() for depth, value in self._b.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LinUCBBandit":
        candidate_depths = tuple(int(value) for value in state["candidate_depths"])
        bandit = cls(candidate_depths, int(state["context_dim"]), alpha=float(state["alpha"]))
        a_state = state["a"]
        b_state = state["b"]
        if not isinstance(a_state, dict) or not isinstance(b_state, dict):
            raise ValueError("invalid bandit state")
        for depth in bandit.candidate_depths:
            bandit._a[depth] = np.asarray(a_state[str(depth)], dtype=np.float64)
            bandit._b[depth] = np.asarray(b_state[str(depth)], dtype=np.float64)
        return bandit

    def _validate_context(self, context: np.ndarray) -> np.ndarray:
        value = np.asarray(context, dtype=np.float64)
        if value.shape != (self.context_dim,):
            raise ValueError(f"context must have shape ({self.context_dim},), got {value.shape}")
        return value
