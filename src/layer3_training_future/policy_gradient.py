"""Policy-gradient controller scaffold for APEX.

The bandit controller should be the first runnable controller. This module is
for the later RL stage, where a neural policy chooses depth and optionally
regime actions from state features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _require_torch():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise RuntimeError("torch is required for policy-gradient controller training") from exc
    return torch, nn, F


@dataclass(frozen=True)
class PolicyConfig:
    state_size: int
    candidate_depths: tuple[int, ...] = (1, 2, 4, 8, 16)
    hidden_size: int = 128


def build_depth_policy(config: PolicyConfig):
    """Return a categorical policy over candidate depths."""

    _torch, nn, _F = _require_torch()

    class DepthPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.candidate_depths = config.candidate_depths
            self.net = nn.Sequential(
                nn.Linear(config.state_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, len(config.candidate_depths)),
            )

        def forward(self, state):
            return self.net(state)

        def sample(self, state):
            torch, _nn, F = _require_torch()
            logits = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
            depths = torch.as_tensor(self.candidate_depths, device=state.device)
            return depths[action_idx], dist.log_prob(action_idx), dist.entropy()

    return DepthPolicy()


def reinforce_loss(log_probs, rewards, *, entropy=None, entropy_coef: float = 0.0):
    """REINFORCE loss with optional entropy bonus."""

    torch, _nn, _F = _require_torch()
    rewards = torch.as_tensor(rewards, dtype=log_probs.dtype, device=log_probs.device)
    advantages = rewards - rewards.mean()
    loss = -(log_probs * advantages.detach()).mean()
    if entropy is not None and entropy_coef:
        loss = loss - entropy_coef * entropy.mean()
    return loss


def clipped_ppo_loss(new_log_probs, old_log_probs, advantages, *, clip_ratio: float = 0.2):
    """Minimal PPO clipped objective for depth-policy updates."""

    torch, _nn, _F = _require_torch()
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()
