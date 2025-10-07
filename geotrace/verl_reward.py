"""verl 0.5.x custom reward function: strict JSON traces and physics gating."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from geotrace.config import RuntimeConfig
from geotrace.constants import INCOMPLETE_REWARD, INVALID_REWARD, OUTCOME_WEIGHT, PROCESS_WEIGHT
from geotrace.data import features, parse_task
from geotrace.process_rewards import (
    bounded_reward,
    monte_carlo_label,
    prefixes_from_trace,
    verify_tool_output,
)
from geotrace.types import JsonRecord

if TYPE_CHECKING:
    from .prm_metrics import ProcessRewardHead
    from .text_prm_model import TextRewardScorer


@lru_cache(maxsize=1)
def _head(path: str | Path) -> tuple[ProcessRewardHead, JsonRecord]:
    from geotrace.train_prm import load_head

    return load_head(path)


@lru_cache(maxsize=1)
def _text_head(path: str | Path) -> TextRewardScorer:
    from geotrace.text_prm import TextRewardScorer

    return TextRewardScorer(path, device=RuntimeConfig.from_env().prm_device)


def _agent_result(score: float, **values: float) -> dict[str, float]:
    # verl's validation requires every auxiliary field on every rollout.
    result = dict.fromkeys(
        (
            "physics_valid",
            "outcome_success",
            "process_score",
            "process_min",
            "process_sum",
            "process_count",
            "reasoning_steps",
            "submitted",
        ),
        0.0,
    )
    return {"score": score, **result, **values}


def _assignment_result(score: float, **values: float) -> dict[str, float]:
    return {
        "score": score,
        "physics_valid": 0.0,
        "format_valid": 0.0,
        "outcome_success": 0.0,
        "process_score": 0.0,
        **values,
    }


def compute_agent_score(
    data_source: str, solution_str: str, ground_truth: object, extra_info: JsonRecord | None = None
) -> dict[str, float]:
    """Score an actual multi-turn tool trajectory using trusted survey data."""
    from geotrace.agent_loop import parse_protocol

    if data_source != "geotrace" or not isinstance(extra_info, dict) or "task" not in extra_info:
        raise ValueError("expected trusted GeoTrace task metadata")
    mode = RuntimeConfig.from_env().reward_mode
    if mode not in {"process", "outcome"}:
        raise ValueError("GEOTRACE_REWARD_MODE must be process or outcome")
    checkpoint = os.environ.get("GEOTRACE_TEXT_PRM")
    if mode == "process" and not checkpoint:
        raise RuntimeError(
            "set GEOTRACE_TEXT_PRM to a trained text PRM, or choose GEOTRACE_REWARD_MODE=outcome"
        )
    task = parse_task(extra_info["task"])
    try:
        steps = parse_protocol(extra_info["task"], solution_str)
        submitted = steps[-1]["tool_call"]["name"] == "submit_model"
        outcome = float(submitted and task.successful(steps[-1]["tool_call"]["arguments"]["model"]))
    except (ValueError, KeyError, TypeError, IndexError):
        return _agent_result(INVALID_REWARD)
    if mode == "outcome":
        return _agent_result(
            outcome,
            physics_valid=1.0,
            outcome_success=outcome,
            submitted=float(submitted),
            reasoning_steps=float(len(steps)),
        )
    # A broken reward model must stop the run, not become a policy penalty.
    probabilities = np.asarray(_text_head(checkpoint).probabilities(extra_info["task"], steps))
    if (
        probabilities.shape != (len(steps),)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise RuntimeError("invalid PRM probability vector. check the trained checkpoint")
    process = float(np.mean(probabilities))
    score = bounded_reward(
        OUTCOME_WEIGHT * outcome + PROCESS_WEIGHT * process if submitted else INCOMPLETE_REWARD,
        physics_valid=True,
    )
    return _agent_result(
        score,
        physics_valid=1.0,
        outcome_success=outcome,
        process_score=process,
        process_min=float(np.min(probabilities)),
        process_sum=float(probabilities.sum()),
        process_count=float(len(probabilities)),
        reasoning_steps=float(len(steps)),
        submitted=float(submitted),
    )


def compute_score(
    data_source: str, solution_str: str, ground_truth: object, extra_info: JsonRecord | None = None
) -> dict[str, float]:
    """Return scalar process score through verl's naive reward manager.

    The response format is intentionally strict. All task data comes
    from the dataset extra_info, never from an untrusted rollout. An optional
    GEOTRACE_PRM_CHECKPOINT replaces MC scoring with the trained process head.
    KL regularization is applied by verl's actor loss, not double-counted here.
    """
    if data_source != "geotrace":
        raise ValueError("unexpected data_source for the GeoTrace reward callable")
    if not isinstance(extra_info, dict) or "task" not in extra_info:
        raise ValueError("the trusted dataset must provide extra_info.task")
    mode = RuntimeConfig.from_env().reward_mode
    task = parse_task(extra_info["task"])
    try:
        answer = json.loads(solution_str)
        if not isinstance(answer, dict) or set(answer) != {
            "assignments",
            "claimed_prediction_microgal",
        }:
            raise ValueError("response must contain the documented JSON fields")
        assignments = answer["assignments"]
        if not isinstance(assignments, list) or len(assignments) != 3:
            raise ValueError("response must assign all three parameters")
        prefixes = prefixes_from_trace(task, assignments)
        valid = verify_tool_output(
            task, prefixes[-1], answer["claimed_prediction_microgal"], relative_tolerance=1e-4
        )
        if not valid:
            return _assignment_result(INVALID_REWARD, format_valid=1.0)
        outcome = float(task.successful(prefixes[-1]))
        if mode == "outcome":
            return _assignment_result(
                outcome, physics_valid=1.0, format_valid=1.0, outcome_success=outcome
            )
        checkpoint = os.environ.get("GEOTRACE_PRM_CHECKPOINT")
        probabilities: list[float]
        if checkpoint:
            import torch

            model, state = _head(checkpoint)
            inputs = torch.tensor(
                [features(task, prefix) for prefix in prefixes], dtype=torch.float32
            )
            with torch.no_grad():
                values = torch.sigmoid(model((inputs - state["mean"]) / state["std"])).numpy()
                probabilities = [float(value) for value in values]
        else:
            # Fixed seed makes reward deterministic for identical traces/tasks.
            rng = np.random.default_rng(17)
            probabilities = [
                monte_carlo_label(task, p, samples=64, rng=rng).probability for p in prefixes
            ]
        score = bounded_reward(
            PROCESS_WEIGHT * float(np.mean(probabilities)) + OUTCOME_WEIGHT * outcome,
            physics_valid=valid,
        )
        return _assignment_result(
            score,
            physics_valid=1.0,
            format_valid=1.0,
            outcome_success=outcome,
            process_score=float(np.mean(probabilities)),
        )
    except (ValueError, KeyError, TypeError, IndexError):
        # Malformed model output is a negative rollout, not a training crash.
        return _assignment_result(INVALID_REWARD)
