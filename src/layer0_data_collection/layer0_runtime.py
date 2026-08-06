"""Layer 0 runtime helpers for APEX survival logging.

This module is intentionally dependency-light so it can be imported from the
existing Phase 2 benchmark runner. It converts vLLM speculative-decoding
positional counters into request-level survival statistics.

Important convention:
- vLLM's `num_accepted_tokens_per_pos[j] / num_drafts` is interpreted as
  S_j = P(positions 0..j all accepted), i.e. survival through position j.
- p_0 = S_0 is the first-token conditional acceptance probability.
- p_j = S_j / S_{j-1} for j > 0.
- h_j = 1 - p_j.
- P(L=0) = 1 - S_0; P(L=j) = S_{j-1} - S_j for j >= 1.
- If all positions 0..k-1 are observed, P(L=k) = S_{k-1}.

This is request-level Layer 0 logging. Strict block-level Layer 0 additionally
requires an internal vLLM accept/reject mask per speculative block. Public
vLLM metrics usually expose aggregate counters, not individual block masks.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

_POS_RE = re.compile(r"(?:accept_rate_pos_|survival_pos_)(\d+)$")
_COUNT_RE = re.compile(r"accept_count_pos_(\d+)$")


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def collect_survival_positions(record: Mapping[str, Any], *, max_depth: int | None = None) -> dict[int, float]:
    """Collect observed survival S_j from `accept_rate_pos_j`/`survival_pos_j` fields."""
    positions: dict[int, float] = {}
    for key, value in record.items():
        match = _POS_RE.match(str(key))
        if not match:
            continue
        idx = int(match.group(1))
        if max_depth is not None and idx >= max_depth:
            continue
        val = _finite_float(value)
        if val is None:
            continue
        positions[idx] = _clip01(val)
    return dict(sorted(positions.items()))


def collect_accept_counts(record: Mapping[str, Any], *, max_depth: int | None = None) -> dict[int, float]:
    """Collect raw accepted-token counts per draft position, if available."""
    counts: dict[int, float] = {}
    for key, value in record.items():
        match = _COUNT_RE.match(str(key))
        if not match:
            continue
        idx = int(match.group(1))
        if max_depth is not None and idx >= max_depth:
            continue
        val = _finite_float(value)
        if val is None:
            continue
        counts[idx] = val
    return dict(sorted(counts.items()))


def survival_to_conditional_and_pmf(survival: Mapping[int, float], *, draft_len: int | None = None) -> dict[str, float]:
    """Convert survival S_j into p_j, h_j, and accepted-length PMF components.

    If `draft_len` is supplied and all positions 0..draft_len-1 are observed,
    the full-acceptance event P(L=draft_len)=S_{draft_len-1} is included.
    Otherwise the final survival value is reported as an observed tail mass:
    P(L >= n_observed) = S_{n_observed-1}.
    """
    if not survival:
        return {
            "apex_observed_positions": 0.0,
            "apex_expected_len_survival_observed": float("nan"),
            "apex_survival_monotonic_violations": 0.0,
        }

    idxs = sorted(survival)
    # Keep contiguous prefix only; if vLLM emits holes, stop at the first gap.
    contiguous: list[int] = []
    for expected_idx in range(max(idxs) + 1):
        if expected_idx not in survival:
            break
        contiguous.append(expected_idx)
    values = [_clip01(survival[i]) for i in contiguous]
    n = len(values)

    out: dict[str, float] = {
        "apex_observed_positions": float(n),
        "apex_expected_len_survival_observed": float(sum(values)),
    }

    violations = 0
    prev = 1.0
    for j, sj in enumerate(values):
        out[f"survival_pos_{j}"] = sj
        if sj > prev + 1e-9:
            violations += 1
        pj = sj / prev if prev > 1e-12 else 0.0
        pj = _clip01(pj)
        out[f"conditional_accept_pos_{j}"] = pj
        out[f"hazard_pos_{j}"] = 1.0 - pj
        # PMF of exact accepted length.
        if j == 0:
            out["pmf_L_eq_0"] = 1.0 - sj
        else:
            out[f"pmf_L_eq_{j}"] = max(0.0, prev - sj)
        prev = sj

    out["apex_survival_monotonic_violations"] = float(violations)
    out["apex_observed_tail_mass_L_ge_n"] = values[-1]

    if draft_len is not None and n >= int(draft_len) and int(draft_len) > 0:
        k = int(draft_len)
        full_mass = values[k - 1]
        out[f"pmf_L_eq_{k}"] = full_mass
        pmf_mass = sum(out.get(f"pmf_L_eq_{j}", 0.0) for j in range(k + 1))
        out["apex_pmf_mass_for_depth"] = pmf_mass
    else:
        # Only a partial PMF is known. The final tail means L >= n_observed.
        out["apex_pmf_mass_observed_prefix_plus_tail"] = 1.0

    return out


def enrich_record_with_layer0_survival(record: Mapping[str, Any], *, max_depth: int | None = None) -> dict[str, Any]:
    """Return a copy of `record` with APEX Layer 0 survival fields added."""
    out: dict[str, Any] = dict(record)
    draft_len = _finite_float(out.get("k", out.get("draft_len")))
    draft_len_int = int(draft_len) if draft_len is not None and draft_len > 0 else max_depth

    survival = collect_survival_positions(out, max_depth=max_depth or draft_len_int)
    counts = collect_accept_counts(out, max_depth=max_depth or draft_len_int)
    for idx, count in counts.items():
        out[f"accept_count_pos_{idx}"] = count
    converted = survival_to_conditional_and_pmf(survival, draft_len=draft_len_int)
    out.update(converted)

    # Counter-based expected length if vLLM exposes accepted_tokens / num_drafts.
    accepted_per_draft = _finite_float(out.get("accepted_per_draft"))
    accepted_total = _finite_float(out.get("accepted_tokens_total", out.get("accepted_tokens")))
    num_drafts = _finite_float(out.get("num_drafts"))
    if accepted_per_draft is not None:
        out["apex_expected_len_counter"] = accepted_per_draft
    elif accepted_total is not None and num_drafts is not None and num_drafts > 0:
        out["apex_expected_len_counter"] = accepted_total / num_drafts
    else:
        out["apex_expected_len_counter"] = float("nan")

    obs = _finite_float(out.get("apex_expected_len_survival_observed"))
    ctr = _finite_float(out.get("apex_expected_len_counter"))
    if obs is not None and ctr is not None:
        out["apex_tail_expected_missing"] = ctr - obs
    else:
        out["apex_tail_expected_missing"] = float("nan")

    out["apex_trace_level"] = "request_aggregate"
    out["apex_strict_block_trace"] = False
    out["apex_layer0_schema_version"] = "0.2-request-survival"
    return out


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values and NaNs into JSON-safe objects."""
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def append_layer0_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one Layer 0 request-level trace row to JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(dict(record)), sort_keys=True) + "\n")
