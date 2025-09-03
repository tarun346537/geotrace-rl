"""Shared numerical types and interfaces at model/backend boundaries."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

JsonRecord: TypeAlias = dict[str, Any]
FloatArray: TypeAlias = NDArray[np.float64]
TokenGenerator: TypeAlias = Callable[[list[int], JsonRecord], Awaitable[list[int]]]


class RewardScorer(Protocol):
    """A process model returns one probability for each supplied prefix."""

    def probabilities(self, task: JsonRecord, steps: list[JsonRecord]) -> list[float]:
        """Score each observed prefix without inspecting hidden labels."""
        ...


class ToolPolicy(Protocol):
    """Model-server contract used by collection and Monte Carlo continuation."""

    model: str

    def next_call(
        self, messages: list[JsonRecord], tools: list[JsonRecord], seed: int
    ) -> JsonRecord:
        """Return one sampled tool action for the supplied chat history."""
        ...


class Tokenizer(Protocol):
    """Only the token operations required by the multi-turn rollout engine."""

    def apply_chat_template(
        self, messages: list[JsonRecord], *, add_generation_prompt: bool, tokenize: bool
    ) -> list[int]:
        """Encode the model's initial conversation and assistant prefix."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode newly introduced environment text only."""
        ...

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        """Read sampled action text without altering the stored token IDs."""
        ...
