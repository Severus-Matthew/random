"""Neural hazard model and censored survival loss."""

from __future__ import annotations

from dataclasses import dataclass


def _require_torch():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise RuntimeError("torch is required for neural hazard training") from exc
    return torch, nn, F


@dataclass(frozen=True)
class HazardModelConfig:
    state_size: int
    regime_size: int
    max_depth: int
    hidden_size: int = 128


def build_hazard_mlp(config: HazardModelConfig):
    """Return h_omega(j | s, r) as a vector of hazards for j=0..max_depth-1."""

    _torch, nn, _F = _require_torch()

    class HazardMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(config.state_size + config.regime_size, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.max_depth),
            )

        def forward(self, state, regime):
            return self.net(torch.cat([state, regime], dim=-1)).sigmoid()

    torch, _nn, _F = _require_torch()
    return HazardMLP()


def censored_hazard_nll(hazards, first_rejection, draft_len):
    """Negative log-likelihood for Equations (7)-(8).

    hazards: tensor [batch, max_depth] with h_j in (0, 1)
    first_rejection: tensor [batch], where R=k denotes full acceptance
    draft_len: tensor [batch]
    """

    torch, _nn, F = _require_torch()
    eps = 1e-7
    hazards = hazards.clamp(eps, 1.0 - eps)
    batch_size, max_depth = hazards.shape
    positions = torch.arange(max_depth, device=hazards.device)[None, :]
    at_risk = positions < draft_len[:, None]
    survived = positions < first_rejection[:, None]
    rejected_here = positions == first_rejection[:, None]

    log_survival = torch.where(survived & at_risk, torch.log1p(-hazards), torch.zeros_like(hazards)).sum(dim=1)
    rejection_log_prob = torch.where(rejected_here & at_risk, torch.log(hazards), torch.zeros_like(hazards)).sum(dim=1)
    return -(log_survival + rejection_log_prob).mean()
