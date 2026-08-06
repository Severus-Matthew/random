"""Data structures for acceptance-utility drafter training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DraftTrainingExample:
    """One supervised drafter-training example derived from a verified block."""

    input_ids: list[int]
    target_ids: list[int]
    regime: list[float]
    draft_len: int
    first_rejection: int
    accept_probs: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.draft_len < 0:
            raise ValueError("draft_len must be non-negative")
        if len(self.target_ids) < self.draft_len:
            raise ValueError("target_ids must contain at least draft_len tokens")
        if not 0 <= self.first_rejection <= self.draft_len:
            raise ValueError("first_rejection must lie in [0, draft_len]")
        if self.accept_probs is not None and len(self.accept_probs) != self.draft_len:
            raise ValueError("accept_probs must have length draft_len")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DraftTrainingExample":
        return cls(**value)


def read_training_jsonl(path: str | Path) -> list[DraftTrainingExample]:
    examples: list[DraftTrainingExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                examples.append(DraftTrainingExample.from_mapping(json.loads(stripped)))
            except Exception as exc:  # pragma: no cover - preserves line context
                raise ValueError(f"invalid training example at {path}:{line_no}: {exc}") from exc
    return examples


def write_training_jsonl(path: str | Path, examples: Iterable[DraftTrainingExample]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.to_json())
            handle.write("\n")
