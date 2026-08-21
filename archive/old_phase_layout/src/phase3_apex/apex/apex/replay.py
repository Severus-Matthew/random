"""Offline replay of the APEX depth controller on block traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from apex.controllers import CostModel, SurvivalDepthController
from apex.hazard import BetaHazardEstimator
from apex.traces import BlockTrace


@dataclass(frozen=True)
class ReplayDecision:
    request_id: str
    block_id: int
    observed_draft_len: int
    observed_accepted_len: int
    chosen_depth_before_update: int
    expected_len_before_update: float
    utility_before_update: float
    accept_probs_before_update: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def replay_depth_controller(
    traces: list[BlockTrace],
    *,
    max_depth: int,
    candidate_depths: tuple[int, ...] = (1, 2, 4, 8, 16),
    alpha: float = 1.0,
    beta: float = 1.0,
    cost_model: CostModel | None = None,
) -> list[ReplayDecision]:
    """Replay online estimation: choose k from current posterior, then update with observed trace."""

    estimator = BetaHazardEstimator(max_depth=max_depth, alpha=alpha, beta=beta)
    controller = SurvivalDepthController(candidate_depths, cost_model=cost_model or CostModel())
    decisions: list[ReplayDecision] = []

    for trace in traces:
        estimate = estimator.estimate(max_depth)
        decision = controller.choose(estimate.accept_probs)
        decisions.append(
            ReplayDecision(
                request_id=trace.request_id,
                block_id=trace.block_id,
                observed_draft_len=trace.draft_len,
                observed_accepted_len=trace.accepted_len,
                chosen_depth_before_update=decision.depth,
                expected_len_before_update=decision.expected_len,
                utility_before_update=decision.utility,
                accept_probs_before_update=decision.accept_probs,
            )
        )
        estimator.update(trace)

    return decisions
