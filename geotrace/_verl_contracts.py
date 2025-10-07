"""Type-only contracts for the pinned verl agent API. Runtime uses verl itself."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import JsonRecord, Tokenizer


class _RolloutSettings(Protocol):
    """Rollout fields consumed by the tool loop."""

    response_length: int


class _ActorSettings(Protocol):
    """The actor's rollout configuration branch."""

    rollout: _RolloutSettings


class _TrainerSettings(Protocol):
    """The trainer branch supplied to an instantiated verl agent."""

    actor_rollout_ref: _ActorSettings


class _ServerManager(Protocol):
    """The pinned server manager's token-in/token-out call."""

    async def generate(
        self, *, request_id: str, prompt_ids: list[int], sampling_params: JsonRecord
    ) -> list[int]:
        """Sample tokens from the selected rollout server."""
        ...


class AgentLoopBase:
    """Attributes supplied by verl 0.5.0 when Hydra creates an agent instance."""

    _class_initialized = False
    config: _TrainerSettings
    server_manager: _ServerManager
    tokenizer: Tokenizer


@dataclass
class AgentLoopOutput:
    """Static field contract of verl's runtime Pydantic output model."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int
    metrics: dict[str, float]
