"""Experiment metrics computed from speculative-decoding block traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean

from apex.hazard import BetaHazardEstimator
from apex.traces import BlockTrace


@dataclass(frozen=True)
class TraceSummary:
    num_blocks: int
    total_drafted_tokens: int
    total_accepted_tokens: int
    acceptance_rate: float
    accepted_tokens_per_verifier_pass: float
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    mean_accepted_len: float
    mean_draft_len: float
    full_acceptance_rate: float
    per_position_acceptance: tuple[float, ...]
    per_position_hazard: tuple[float, ...]
    per_position_exposures: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def summarize_traces(traces: list[BlockTrace], *, max_depth: int | None = None) -> TraceSummary:
    if not traces:
        return TraceSummary(
            num_blocks=0,
            total_drafted_tokens=0,
            total_accepted_tokens=0,
            acceptance_rate=0.0,
            accepted_tokens_per_verifier_pass=0.0,
            mean_latency_ms=None,
            p95_latency_ms=None,
            mean_accepted_len=0.0,
            mean_draft_len=0.0,
            full_acceptance_rate=0.0,
            per_position_acceptance=(),
            per_position_hazard=(),
            per_position_exposures=(),
        )

    inferred_max_depth = max(trace.draft_len for trace in traces)
    estimator = BetaHazardEstimator(max_depth or inferred_max_depth, alpha=1.0, beta=1.0)
    estimator.update_many(traces)
    estimate = estimator.estimate()

    total_drafted = sum(trace.draft_len for trace in traces)
    total_accepted = sum(trace.accepted_len for trace in traces)
    latencies = [trace.latency_ms for trace in traces if trace.latency_ms is not None]
    return TraceSummary(
        num_blocks=len(traces),
        total_drafted_tokens=total_drafted,
        total_accepted_tokens=total_accepted,
        acceptance_rate=(total_accepted / total_drafted) if total_drafted else 0.0,
        accepted_tokens_per_verifier_pass=total_accepted / len(traces),
        mean_latency_ms=mean(latencies) if latencies else None,
        p95_latency_ms=_percentile(latencies, 0.95),
        mean_accepted_len=total_accepted / len(traces),
        mean_draft_len=total_drafted / len(traces),
        full_acceptance_rate=sum(trace.fully_accepted for trace in traces) / len(traces),
        per_position_acceptance=estimate.accept_probs,
        per_position_hazard=estimate.hazards,
        per_position_exposures=estimate.exposures,
    )
