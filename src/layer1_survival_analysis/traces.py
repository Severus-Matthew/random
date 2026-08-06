"""Trace schema for APEX speculative-decoding blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BlockTrace:
    """One verified speculative block.

    first_rejection is zero-based. If all drafted tokens are accepted, set
    first_rejection == draft_len. Under this convention accepted_len equals
    first_rejection.
    """

    request_id: str
    block_id: int
    method: str
    draft_len: int
    first_rejection: int
    latency_ms: float | None = None
    output_tokens: int | None = None
    entropy: float | None = None
    repetition_density: float | None = None
    temperature: float | None = None
    target_drift: float | None = None
    regime: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.draft_len < 0:
            raise ValueError("draft_len must be non-negative")
        if not 0 <= self.first_rejection <= self.draft_len:
            raise ValueError("first_rejection must lie in [0, draft_len]")
        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")

    @property
    def accepted_len(self) -> int:
        return self.first_rejection

    @property
    def fully_accepted(self) -> bool:
        return self.first_rejection == self.draft_len

    def observed_label(self, position: int) -> int | None:
        """Return 1 accepted, 0 rejected, or None if unobserved at position."""

        if position < 0:
            raise ValueError("position must be non-negative")
        if position >= self.draft_len:
            return None
        if self.first_rejection > position:
            return 1
        if self.first_rejection == position:
            return 0
        return None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BlockTrace":
        return cls(**value)


def read_jsonl(path: str | Path) -> list[BlockTrace]:
    traces: list[BlockTrace] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                traces.append(BlockTrace.from_mapping(json.loads(stripped)))
            except Exception as exc:  # pragma: no cover - preserves line context
                raise ValueError(f"invalid trace at {path}:{line_no}: {exc}") from exc
    return traces


def write_jsonl(path: str | Path, traces: Iterable[BlockTrace]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(trace.to_json())
            handle.write("\n")
