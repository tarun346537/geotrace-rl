"""vLLM tool rollouts, early process-reward pruning, and Monte Carlo labels."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict

from .constants import MAX_TURNS, PRUNE_THRESHOLD, SEED, WILSON_Z
from .data import parse_task
from .model_client import InvalidAction as InvalidAction
from .model_client import RolloutError as RolloutError
from .model_client import VLLMClient as VLLMClient
from .physics import GravityTask, relative_rmse
from .process_rewards import ProcessLabel, verify_tool_output
from .types import JsonRecord, RewardScorer, ToolPolicy


def tools_for(task: GravityTask) -> list[JsonRecord]:
    """Describe the two simulator actions and their finite parameter grids."""
    model_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            key: {"type": "number", "enum": list(values)} for key, values in task.grid.items()
        },
        "required": list(task.grid),
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"reasoning": {"type": "string"}, "model": model_schema},
                    "required": ["reasoning", "model"],
                },
            },
        }
        for name, description in (
            (
                "gravity_forward",
                "Compute the specified gravity model with the trusted forward simulator.",
            ),
            ("submit_model", "Submit the final gravity model for independent forward evaluation."),
        )
    ]


def initial_messages(task_dict: JsonRecord) -> list[JsonRecord]:
    """Build a model-visible task without hidden targets or evaluation data."""
    return [
        {
            "role": "system",
            "content": "Infer the gravity source from the supplied observations. Explain each proposed model in the reasoning "
            "argument, call gravity_forward to test it, and finish with submit_model. Use only supplied grid values. "
            "Tool results are supplied by the environment. The task specifies the forward backend and units.",
        },
        {"role": "user", "content": json.dumps(task_dict, sort_keys=True, allow_nan=False)},
    ]


def append_step(messages: list[JsonRecord], step: JsonRecord) -> None:
    """Append verified simulator history in the model-server chat format."""
    call = step["tool_call"]
    call_id = step.get("call_id", f"gravity-{len(messages)}")
    messages.append(
        {
            "role": "assistant",
            "content": step["reasoning"],
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], allow_nan=False),
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(step["tool_result"], allow_nan=False),
        }
    )


def execute_tool(task: GravityTask, name: str, arguments: JsonRecord) -> JsonRecord:
    """Validate model arguments and execute the selected gravity solver."""
    if name not in {"gravity_forward", "submit_model"}:
        raise ValueError("unknown tool")
    if not isinstance(arguments, dict) or set(arguments) != {"reasoning", "model"}:
        raise ValueError("tool arguments require reasoning and model")
    if not isinstance(arguments["reasoning"], str) or not arguments["reasoning"].strip():
        raise ValueError("nonempty reasoning is required")
    prediction = task.prediction(arguments["model"])
    return {
        "prediction_microgal": prediction.tolist(),
        "relative_rmse": relative_rmse(prediction, task.observations_microgal),
        "success": bool(task.successful(arguments["model"])),
        "forward_backend": task.forward_backend,
    }


def verify_steps(task: GravityTask, steps: list[JsonRecord]) -> bool:
    """Recompute tool outputs rather than trusting a rollout's declared success field."""
    for index, step in enumerate(steps):
        try:
            call, result = step["tool_call"], step["tool_result"]
            expected = execute_tool(task, call["name"], call["arguments"])
            if (
                step["reasoning"] != call["arguments"]["reasoning"]
                or set(result) != set(expected)
                or result["forward_backend"] != expected["forward_backend"]
                or (call["name"] == "submit_model" and index != len(steps) - 1)
            ):
                return False
            if not verify_tool_output(
                task, call["arguments"]["model"], result["prediction_microgal"]
            ):
                return False
            if type(result["success"]) is not bool or result["success"] != expected["success"]:
                return False
            error = result["relative_rmse"]
            if (
                not isinstance(error, (int, float))
                or not math.isfinite(error)
                or not math.isclose(error, expected["relative_rmse"], rel_tol=1e-7, abs_tol=1e-10)
            ):
                return False
        except (ValueError, TypeError, KeyError, IndexError):
            return False
    return bool(steps)


def rollout(
    task_dict: JsonRecord,
    client: ToolPolicy,
    *,
    max_turns: int = MAX_TURNS,
    seed: int = SEED,
    prefix: list[JsonRecord] | None = None,
    scorer: RewardScorer | None = None,
    prune_threshold: float = PRUNE_THRESHOLD,
) -> JsonRecord:
    """Execute model tool actions until submission, pruning or a turn limit."""
    if max_turns < 1 or not 0 <= prune_threshold <= 1:
        raise ValueError("positive max_turns and a probability pruning threshold required")
    task = parse_task(task_dict)
    steps = copy.deepcopy(prefix or [])
    if steps and not verify_steps(task, steps):
        raise ValueError("prefix contains inconsistent tool outputs")
    messages = initial_messages(task_dict)
    for step in steps:
        append_step(messages, step)
    if steps and steps[-1]["tool_call"]["name"] == "submit_model":
        return {
            "steps": steps,
            "success": bool(task.successful(steps[-1]["tool_call"]["arguments"]["model"])),
            "termination": "submitted",
        }
    for index in range(len(steps), max_turns):
        try:
            call = client.next_call(messages, tools_for(task), seed + index)
        except InvalidAction:
            return {"steps": steps, "success": False, "termination": "invalid_action"}
        try:
            result = execute_tool(task, call["name"], call["arguments"])
        except (ValueError, KeyError, TypeError):
            return {"steps": steps, "success": False, "termination": "invalid_action"}
        step = {
            "reasoning": call["arguments"]["reasoning"],
            "tool_call": {"name": call["name"], "arguments": call["arguments"]},
            "tool_result": result,
            "call_id": call["id"],
        }
        steps.append(step)
        append_step(messages, step)
        if call["name"] == "submit_model":
            return {"steps": steps, "success": result["success"], "termination": "submitted"}
        if scorer is not None:
            probability = float(scorer.probabilities(task_dict, steps)[-1])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("PRM returned an invalid probability")
            step["process_probability"] = probability
            if probability < prune_threshold:
                return {"steps": steps, "success": False, "termination": "process_reward_pruned"}
    return {"steps": steps, "success": False, "termination": "turn_budget"}


def label_trace(
    trace: JsonRecord,
    client: ToolPolicy,
    *,
    samples: int = 16,
    seed: int = SEED,
    max_turns: int = MAX_TURNS,
) -> JsonRecord:
    """Resample continuations from every prefix, with pruning disabled."""
    if samples < 1:
        raise ValueError("positive Monte Carlo sample count required")
    result = copy.deepcopy(trace)
    task = parse_task(trace["task"])
    if not verify_steps(task, trace["steps"]):
        raise ValueError("cannot label physically inconsistent tool history")
    for index, step in enumerate(result["steps"]):
        successes = 0
        for sample in range(samples):
            continuation = rollout(
                trace["task"],
                client,
                prefix=trace["steps"][: index + 1],
                max_turns=max_turns,
                seed=seed + index * samples * 1009 + sample * 1009,
            )
            successes += int(continuation["success"])
        probability, z = successes / samples, WILSON_Z
        denominator = 1 + z * z / samples
        center = (probability + z * z / (2 * samples)) / denominator
        half = (
            z
            * math.sqrt(probability * (1 - probability) / samples + z * z / (4 * samples * samples))
            / denominator
        )
        step["mc_label"] = asdict(
            ProcessLabel(
                successes, samples, probability, max(0, center - half), min(1, center + half)
            )
        )
    result["labeling"] = {
        "method": "model_policy_monte_carlo",
        "continuation_model": client.model,
        "samples_per_prefix": samples,
        "seed": seed,
        "pruning_during_labeling": False,
    }
    return result


def _label_remote(
    trace: JsonRecord, endpoint: str, model: str, samples: int, seed: int, max_turns: int
) -> JsonRecord:
    return label_trace(
        trace, VLLMClient(endpoint, model), samples=samples, seed=seed, max_turns=max_turns
    )


def main() -> None:
    """Run model collection or process-label commands."""
    from .rollout_cli import main as run

    run()


if __name__ == "__main__":
    main()
