"""Inference from an exported transformer process-reward head."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .text_prm_data import Trace, prefix_text


def inference_settings(
    model_directory: str | Path, max_length: int | None = None
) -> dict[str, Any]:
    """Read the exact context contract used to train an exported head."""
    path = Path(model_directory) / "prm_config.json"
    settings = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if (
        settings.get("truncation_side") != "left"
        or not isinstance(settings.get("max_length"), int)
        or settings["max_length"] < 32
    ):
        raise ValueError("invalid saved PRM context settings. export the trained head again")
    if max_length is not None and max_length != settings["max_length"]:
        raise ValueError("inference context length must match the trained PRM")
    return settings


class TextRewardScorer:
    """Batch prefix scoring with the checkpoint's tokenizer and context limit."""

    def __init__(
        self, model_directory: str | Path, device: str = "cpu", max_length: int | None = None
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        settings = inference_settings(model_directory, max_length)
        self.device, self.max_length = device, int(settings["max_length"])
        tokenizer_loader = cast(
            "Callable[..., PreTrainedTokenizerBase]", AutoTokenizer.from_pretrained
        )
        self.tokenizer: PreTrainedTokenizerBase = tokenizer_loader(model_directory)
        self.tokenizer.truncation_side = str(settings["truncation_side"])
        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(model_directory).to(device).eval()
        )

    def probabilities(self, task: Mapping[str, Any], steps: Sequence[Trace]) -> list[float]:
        """Return one continuation-success probability per observed prefix."""
        import torch

        texts = [prefix_text(task, steps[: i + 1]) for i in range(len(steps))]
        if not texts:
            return []
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = cast(
                torch.Tensor,
                self.model(**{key: value.to(self.device) for key, value in inputs.items()}).logits[
                    :, 0
                ],
            )
        return [float(value) for value in torch.sigmoid(logits).cpu().tolist()]
