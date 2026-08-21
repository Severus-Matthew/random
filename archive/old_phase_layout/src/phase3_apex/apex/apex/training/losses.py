"""Acceptance-utility training weights and regularizers."""

from __future__ import annotations

from collections.abc import Sequence

from apex.survival import survival_curve


def acceptance_utility_weights(
    accept_probs: Sequence[float],
    *,
    first_rejection: int | None = None,
    rejection_bonus: float = 0.0,
    eps: float = 1e-8,
) -> list[float]:
    """Return g_m weights from d E[L|k] / d p_m.

    With the corrected convention:

        E[L|k] = sum_{i=0}^{k-1} S_i,  S_i = prod_{l=0}^i p_l
        g_m = sum_{i=m}^{k-1} S_i / p_m

    If first_rejection is in 0..k-1, that position receives the optional
    multiplicative rejection bonus. If first_rejection == k, the block was
    fully accepted and no drafted position receives the bonus.
    """

    k = len(accept_probs)
    if first_rejection is not None and not 0 <= first_rejection <= k:
        raise ValueError("first_rejection must lie in [0, k]")
    if rejection_bonus < 0.0:
        raise ValueError("rejection_bonus must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    survival = survival_curve(accept_probs)
    weights: list[float] = []
    for position, prob in enumerate(accept_probs):
        utility = sum(survival[position:]) / max(prob, eps)
        if first_rejection == position:
            utility *= 1.0 + rejection_bonus
        weights.append(utility)
    return weights


def flattening_penalty(accept_probs: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Return sum_{m=1}^{k-1} g_m (p_m - p_{m-1})^2."""

    if len(accept_probs) <= 1:
        return 0.0
    if weights is None:
        weights = [1.0] * len(accept_probs)
    if len(weights) != len(accept_probs):
        raise ValueError("weights and accept_probs must have the same length")
    total = 0.0
    for position in range(1, len(accept_probs)):
        delta = accept_probs[position] - accept_probs[position - 1]
        total += weights[position] * delta * delta
    return total


def torch_acceptance_utility_loss(
    log_probs,
    weights,
    *,
    normalize: bool = True,
):
    """Torch loss for -sum_m w_m log q_phi(y_m^T | ...).

    The function accepts torch tensors but imports torch lazily so the package
    remains usable in CPU-only environments without training dependencies.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise RuntimeError("torch is required for torch_acceptance_utility_loss") from exc

    weight_tensor = torch.as_tensor(weights, dtype=log_probs.dtype, device=log_probs.device)
    loss = -(weight_tensor * log_probs).sum()
    if normalize:
        loss = loss / weight_tensor.sum().clamp_min(1e-8)
    return loss
