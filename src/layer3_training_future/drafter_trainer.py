"""Training loop skeleton for acceptance-utility LoRA drafter fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apex.training.data import read_training_jsonl
from apex.training.losses import acceptance_utility_weights


@dataclass(frozen=True)
class DrafterTrainingConfig:
    model_name_or_path: str
    train_jsonl: str
    output_dir: str
    learning_rate: float = 2e-4
    batch_size: int = 1
    num_epochs: int = 1
    max_depth: int = 16
    rejection_bonus: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def build_weighted_labels(examples_path: str, *, rejection_bonus: float = 1.0) -> list[dict[str, object]]:
    """Precompute position weights for a JSONL drafter-training file."""

    rows: list[dict[str, object]] = []
    for example in read_training_jsonl(examples_path):
        if example.accept_probs is None:
            accept_probs = [1.0 if idx < example.first_rejection else 0.5 for idx in range(example.draft_len)]
            if example.first_rejection == example.draft_len:
                accept_probs = [1.0] * example.draft_len
        else:
            accept_probs = example.accept_probs
        rows.append(
            {
                "input_ids": example.input_ids,
                "target_ids": example.target_ids[: example.draft_len],
                "regime": example.regime,
                "weights": acceptance_utility_weights(
                    accept_probs,
                    first_rejection=example.first_rejection,
                    rejection_bonus=rejection_bonus,
                ),
                "metadata": example.metadata,
            }
        )
    return rows


def train_peft_lora_drafter(config: DrafterTrainingConfig) -> None:
    """Fine-tune a causal LM drafter with PEFT LoRA.

    This function is intentionally a thin integration point. The exact batch
    collation depends on the tokenizer and vLLM trace export format, so the
    code validates dependencies and prepares weighted examples, then leaves the
    final Trainer wiring explicit for the experiment environment.
    """

    try:
        import torch  # noqa: F401
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise RuntimeError("torch, transformers, and peft are required for LoRA drafter training") from exc

    from apex.training.conditioning import build_peft_lora_config

    weighted_rows = build_weighted_labels(config.train_jsonl, rejection_bonus=config.rejection_bonus)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    model = get_peft_model(
        model,
        build_peft_lora_config(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        ),
    )

    class WeightedListDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(weighted_rows)

        def __getitem__(self, idx):
            return weighted_rows[idx]

    def collate(batch):
        raise NotImplementedError(
            "Provide tokenizer-specific collation that maps input_ids/target_ids/weights "
            "to model inputs and token-level weighted loss."
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        remove_unused_columns=False,
    )
    _ = tokenizer
    _ = Trainer(model=model, args=training_args, train_dataset=WeightedListDataset(), data_collator=collate)
    raise NotImplementedError(
        "Trainer object is constructed, but the final weighted-loss Trainer subclass "
        "must be wired once the exact tokenizer/vLLM trace format is fixed."
    )
