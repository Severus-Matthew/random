"""Adapter and LoRA conditioning modules for the APEX drafter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise RuntimeError("torch is required for APEX conditioning modules") from exc
    return torch, nn


@dataclass(frozen=True)
class AdapterConfig:
    hidden_size: int
    regime_size: int
    bottleneck_size: int = 64


@dataclass(frozen=True)
class RegimeLoRAConfig:
    in_features: int
    out_features: int
    rank: int = 8
    num_experts: int = 3
    regime_size: int = 8
    alpha: float = 16.0


def build_residual_regime_adapter(config: AdapterConfig):
    """Return h' = h + W2 sigma(W1 [h; r])."""

    torch, nn = _require_torch()

    class ResidualRegimeAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = nn.Linear(config.hidden_size + config.regime_size, config.bottleneck_size)
            self.up = nn.Linear(config.bottleneck_size, config.hidden_size)
            self.activation = nn.SiLU()

        def forward(self, hidden_states, regime):
            if regime.dim() == 2 and hidden_states.dim() == 3:
                regime_expanded = regime[:, None, :].expand(-1, hidden_states.shape[1], -1)
            else:
                regime_expanded = regime
            conditioned = torch.cat([hidden_states, regime_expanded], dim=-1)
            return hidden_states + self.up(self.activation(self.down(conditioned)))

    return ResidualRegimeAdapter()


def build_regime_lora_linear(base_linear, config: RegimeLoRAConfig):
    """Wrap a Linear layer with a regime-dependent mixture of low-rank updates.

    Computes W'(r) = W + sum_m softmax(G r)_m A_m B_m.
    """

    torch, nn = _require_torch()

    class RegimeLoRALinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base_linear
            self.rank = config.rank
            self.scaling = config.alpha / max(1, config.rank)
            self.router = nn.Linear(config.regime_size, config.num_experts, bias=False)
            self.lora_a = nn.Parameter(torch.zeros(config.num_experts, config.rank, config.in_features))
            self.lora_b = nn.Parameter(torch.zeros(config.num_experts, config.out_features, config.rank))
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)

            for parameter in self.base.parameters():
                parameter.requires_grad = False

        def forward(self, inputs, regime):
            base_out = self.base(inputs)
            mix = torch.softmax(self.router(regime), dim=-1)
            if inputs.dim() == 3 and regime.dim() == 2:
                mix = mix[:, None, :]
            low_rank = torch.einsum("...i,eri->...er", inputs, self.lora_a)
            delta = torch.einsum("...er,eor->...eo", low_rank, self.lora_b)
            delta = (delta * mix[..., :, None]).sum(dim=-2)
            return base_out + self.scaling * delta

    return RegimeLoRALinear()


def build_peft_lora_config(
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    task_type: str = "CAUSAL_LM",
):
    """Return a PEFT LoraConfig for standard LoRA drafter fine-tuning."""

    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise RuntimeError("peft is required for build_peft_lora_config") from exc

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        task_type=task_type,
    )
