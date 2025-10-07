"""Token-preserving multi-turn gravity rollouts, independent of a trainer."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .constants import END_STEP, MAX_TURNS, PRUNE_THRESHOLD, RESPONSE_LENGTH, TOOL_CLOSE, TOOL_OPEN
from .data import parse_task
from .physics import GravityTask
from .rollouts import execute_tool, verify_steps
from .types import JsonRecord, RewardScorer, TokenGenerator, Tokenizer

AGENT_INSTRUCTION = (
    "Use the trusted gravity_forward tool to investigate the survey, then submit_model to finish. "
    "At each turn emit exactly one JSON object with keys reasoning (nonempty string), tool "
    "(gravity_forward or submit_model), and model (x_m, depth_m, mass_kg chosen from the task grid). "
    "End each object with a newline and <END_STEP>. The environment inserts TOOL_RESULT observations. "
    "Do not write tool results yourself. Read the backend and units in the survey."
)


def load_pruning_scorer(
    environment: Mapping[str, str], loader: Callable[..., RewardScorer]
) -> RewardScorer | None:
    """Keep the outcome-only ablation independent of any configured PRM."""
    mode = environment.get("GEOTRACE_REWARD_MODE", "process")
    if mode not in {"process", "outcome"}:
        raise ValueError("GEOTRACE_REWARD_MODE must be process or outcome")
    checkpoint = environment.get("GEOTRACE_TEXT_PRM")
    if mode == "outcome" or not checkpoint:
        return None
    return loader(checkpoint, device=environment.get("GEOTRACE_PRM_DEVICE", "cpu"))


def decode_action(text: str) -> JsonRecord:
    """Decode exactly one model action ending with the protocol marker."""
    if END_STEP not in text:
        raise ValueError("generated action is missing END_STEP")
    content, suffix = text.split(END_STEP, 1)
    if suffix.strip():
        raise ValueError("unexpected tokens after END_STEP. emit one action per turn")
    content = content.strip()
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"reasoning", "tool", "model"}:
        raise ValueError("expected one reasoning/tool/model JSON object")
    return value


def action_step(task: GravityTask, action: JsonRecord, index: int) -> JsonRecord:
    """Execute one validated model action against the trusted simulator."""
    arguments = {"reasoning": action["reasoning"], "model": action["model"]}
    result = execute_tool(task, action["tool"], arguments)
    return {
        "reasoning": action["reasoning"],
        "tool_call": {"name": action["tool"], "arguments": arguments},
        "tool_result": result,
        "call_id": f"gravity-step-{index}",
    }


def parse_protocol(task_dict: JsonRecord, text: str) -> list[JsonRecord]:
    """Parse model actions and independently verify any embedded tool results."""
    steps: list[JsonRecord] = []
    task, remainder = parse_task(task_dict), text
    decoder = json.JSONDecoder()
    while remainder.strip():
        remainder = remainder.lstrip()
        action, end = decoder.raw_decode(remainder)
        remainder = remainder[end:]
        if not remainder.startswith(END_STEP):
            raise ValueError("every generated action must end with END_STEP")
        remainder = remainder[len(END_STEP) :]
        if not isinstance(action, dict) or set(action) != {"reasoning", "tool", "model"}:
            raise ValueError("unexpected action fields")
        step = action_step(task, action, len(steps))
        if remainder.startswith(TOOL_OPEN):
            remainder = remainder[len(TOOL_OPEN) :]
            observation, end = decoder.raw_decode(remainder)
            remainder = remainder[end:]
            if not remainder.startswith(TOOL_CLOSE):
                raise ValueError("malformed tool observation boundary")
            remainder = remainder[len(TOOL_CLOSE) :]
            check = dict(step, tool_result=observation)
            if not verify_steps(task, [check]):
                raise ValueError("tool observation violates the trusted forward model")
        steps.append(step)
        if action["tool"] == "submit_model":
            if remainder.strip():
                raise ValueError("no actions may follow final submission")
            break
    if not steps:
        raise ValueError("no tool actions found")
    return steps


@dataclass
class TokenRollout:
    """Sampled token IDs, their loss mask and verified simulator history."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    steps: list[JsonRecord]
    termination: str
    generation_seconds: float
    tool_seconds: float


async def run_token_loop(
    tokenizer: Tokenizer,
    generate: TokenGenerator,
    messages: list[JsonRecord],
    task_dict: JsonRecord,
    sampling_params: JsonRecord,
    *,
    max_turns: int = MAX_TURNS,
    response_length: int = RESPONSE_LENGTH,
    scorer: RewardScorer | None = None,
    prune_threshold: float = PRUNE_THRESHOLD,
) -> TokenRollout:
    """Append generated tokens verbatim, with zero loss mask on observation tokens.

    ``generate`` is an async token-in/token-out callable. Only NEW observation
    text is encoded. Sampled model tokens are never decoded and re-encoded.
    """
    if max_turns < 1 or response_length < 1 or not 0 <= prune_threshold <= 1:
        raise ValueError("invalid rollout limits")
    task = parse_task(task_dict)
    prompt = list(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    )
    response: list[int] = []
    mask: list[int] = []
    steps: list[JsonRecord] = []
    generation_seconds = tool_seconds = 0.0
    termination = "turn_budget"
    params = dict(sampling_params)
    params.update(stop=[END_STEP], include_stop_str_in_output=True)
    for _ in range(max_turns):
        started = time.perf_counter()
        generated = list(await generate(prompt + response, params))
        generation_seconds += time.perf_counter() - started
        if not generated:
            if not response:
                raise RuntimeError("the rollout server returned no initial model tokens")
            termination = "empty_generation"
            break
        remaining = response_length - len(response)
        if not generated or len(generated) > remaining:
            response.extend(generated[:remaining])
            mask.extend([1] * min(len(generated), remaining))
            termination = "response_budget"
            break
        response.extend(generated)
        mask.extend([1] * len(generated))
        try:
            action = decode_action(tokenizer.decode(generated, skip_special_tokens=True))
            started = time.perf_counter()
            step = await asyncio.to_thread(action_step, task, action, len(steps))
            tool_seconds += time.perf_counter() - started
        except (ValueError, KeyError, TypeError):
            termination = "invalid_action"
            break
        steps.append(step)
        if action["tool"] == "submit_model":
            termination = "submitted"
            break
        if scorer is not None:
            probability = float(
                (await asyncio.to_thread(scorer.probabilities, task_dict, steps))[-1]
            )
            if not 0 <= probability <= 1:
                raise ValueError("invalid PRM probability")
            step["process_probability"] = probability
            if probability < prune_threshold:
                termination = "process_reward_pruned"
                break
        if len(steps) >= max_turns:
            break
        observation = TOOL_OPEN + json.dumps(step["tool_result"], allow_nan=False) + TOOL_CLOSE
        observation_ids = list(tokenizer.encode(observation, add_special_tokens=False))
        # Preserve one model-token position for the next action.
        if len(response) + len(observation_ids) >= response_length:
            termination = "response_budget"
            break
        response.extend(observation_ids)
        mask.extend([0] * len(observation_ids))
    # verl's scalar reward manager attaches the terminal score to the final
    # attention-valid token. Keep it on a model token for PPO/GAE as well as GRPO.
    while mask and mask[-1] == 0:
        mask.pop()
        response.pop()
    return TokenRollout(
        prompt, response, mask, steps, termination, generation_seconds, tool_seconds
    )
