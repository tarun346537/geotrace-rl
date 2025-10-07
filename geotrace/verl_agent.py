"""verl==0.5.0 adapter for the token-preserving gravity tool loop."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from ._verl_contracts import AgentLoopBase, AgentLoopOutput
else:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from geotrace.agent_loop import load_pruning_scorer, run_token_loop
from geotrace.constants import MAX_TURNS, PRUNE_THRESHOLD
from geotrace.types import JsonRecord, RewardScorer, Tokenizer


class GravityAgentLoop(AgentLoopBase):
    """Execute trusted gravity tools between token-preserving policy generations."""

    scorer: RewardScorer | None = None
    max_turns: int = MAX_TURNS
    prune_threshold: float = PRUNE_THRESHOLD

    @classmethod
    def init_class(cls, config: object, tokenizer: Tokenizer, **kwargs: object) -> None:
        """Load shared limits and an optional process model once per agent worker."""
        if cls._class_initialized:
            return
        cls.max_turns = int(str(kwargs.get("max_turns", MAX_TURNS)))
        cls.prune_threshold = float(str(kwargs.get("prune_threshold", PRUNE_THRESHOLD)))
        from geotrace.text_prm import TextRewardScorer

        cls.scorer = load_pruning_scorer(os.environ, TextRewardScorer)
        cls._class_initialized = True

    async def run(self, messages: list[JsonRecord], sampling_params: JsonRecord) -> AgentLoopOutput:
        """Return sampled IDs and a mask excluding environment observation tokens."""
        task = json.loads(messages[-1]["content"])
        request_id = uuid4().hex

        async def _generate(prompt_ids: list[int], parameters: JsonRecord) -> list[int]:
            return await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=parameters
            )

        result = await run_token_loop(
            self.tokenizer,
            _generate,
            messages,
            task,
            sampling_params,
            max_turns=self.max_turns,
            response_length=self.config.actor_rollout_ref.rollout.response_length,
            scorer=self.scorer,
            prune_threshold=self.prune_threshold,
        )
        return AgentLoopOutput(
            prompt_ids=result.prompt_ids,
            response_ids=result.response_ids,
            response_mask=result.response_mask,
            num_turns=len(result.steps),
            metrics={
                "generate_sequences": result.generation_seconds,
                "tool_calls": result.tool_seconds,
            },
        )
