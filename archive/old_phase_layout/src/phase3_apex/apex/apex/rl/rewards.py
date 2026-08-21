"""Reward functions for APEX controller training."""

from __future__ import annotations

from dataclasses import dataclass

from apex.traces import BlockTrace


@dataclass(frozen=True)
class RewardConfig:
    waste_penalty: float = 0.05
    tail_latency_penalty: float = 0.0
    target_latency_ms: float | None = None
    switch_penalty: float = 0.0


def block_reward(
    trace: BlockTrace,
    *,
    config: RewardConfig | None = None,
    previous_depth: int | None = None,
) -> float:
    """Reward accepted progress per latency, with waste and tail penalties."""

    cfg = config or RewardConfig()
    latency_ms = trace.latency_ms if trace.latency_ms is not None and trace.latency_ms > 0 else 1.0
    progress = trace.accepted_len / latency_ms
    wasted = max(0, trace.draft_len - trace.accepted_len)
    reward = progress - cfg.waste_penalty * wasted
    if cfg.target_latency_ms is not None and trace.latency_ms is not None:
        reward -= cfg.tail_latency_penalty * max(0.0, trace.latency_ms - cfg.target_latency_ms)
    if previous_depth is not None:
        reward -= cfg.switch_penalty * abs(trace.draft_len - previous_depth)
    return reward
