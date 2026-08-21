"""APEX core utilities for adaptive speculative decoding experiments."""

from apex.survival import (
    accepted_length_pmf,
    expected_accepted_length,
    expected_accepted_length_from_hazards,
    marginal_depth_value,
    survival_curve,
)

__all__ = [
    "accepted_length_pmf",
    "expected_accepted_length",
    "expected_accepted_length_from_hazards",
    "marginal_depth_value",
    "survival_curve",
]
