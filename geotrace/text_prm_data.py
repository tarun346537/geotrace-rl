"""Text-prefix schema checks and first-error measurements."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from .survey_identity import SurveyRegistry

Trace = dict[str, Any]


def prefix_text(task: Mapping[str, Any], steps: Sequence[Trace]) -> str:
    """Render observable history without MC targets or error annotations."""
    visible = [
        {key: step[key] for key in ("reasoning", "tool_call", "tool_result") if key in step}
        for step in steps
    ]
    survey = json.dumps(dict(task), sort_keys=True, allow_nan=False)
    history = json.dumps(visible, sort_keys=True, allow_nan=False)
    return f"Gravity survey:\n{survey}\nReasoning and tool history:\n{history}"


def load_traces(path: str | Path) -> list[Trace]:
    """Read tool traces and reject duplicate identities or split leakage."""
    from .data import parse_task

    traces: list[Trace] = []
    ids: set[str] = set()
    surveys = SurveyRegistry()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            trace = cast(Trace, json.loads(line))
            trace_id, survey_id, split = trace["trace_id"], trace["survey_id"], trace["split"]
            if not isinstance(trace_id, str) or not trace_id or trace_id in ids:
                raise ValueError("trace_id must be nonempty and unique")
            if (
                not isinstance(survey_id, str)
                or not survey_id
                or split not in {"train", "validation", "test"}
            ):
                raise ValueError("supply survey_id and a train/validation/test split")
            surveys.add(survey_id, split, parse_task(trace["task"]))
            if not isinstance(trace["steps"], list) or not trace["steps"]:
                raise ValueError("at least one reasoning step is required")
            for step in trace["steps"]:
                if not isinstance(step.get("reasoning"), str) or not step["reasoning"].strip():
                    raise ValueError("each step needs nonempty reasoning text")
                if not isinstance(step.get("tool_call"), dict) or not isinstance(
                    step.get("tool_result"), dict
                ):
                    raise ValueError("each step needs a structured tool call and result")
                if "mc_label" in step:
                    probability = float(step["mc_label"]["probability"])
                    if not np.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError("MC target must be a finite probability. rerun labeling")
                if "error_label" in step and type(step["error_label"]) is not bool:
                    raise ValueError("independent error_label must be boolean")
            ids.add(trace_id)
            traces.append(trace)
        except (KeyError, ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"{path}:{number}: invalid trace: {error}") from error
    if not traces:
        raise ValueError("no traces found. check the input JSONL path")
    return traces


def examples(traces: Sequence[Trace], require_labels: bool = True) -> dict[str, list[Trace]]:
    """Expand each trace into labeled prefixes while preserving its survey split."""
    from .data import parse_task
    from .rollouts import verify_steps

    splits: dict[str, list[Trace]] = {name: [] for name in ("train", "validation", "test")}
    for trace in traces:
        if require_labels and not verify_steps(parse_task(trace["task"]), trace["steps"]):
            raise ValueError(
                "training trace has inconsistent tool history. rerun collection and labeling"
            )
        for index, step in enumerate(trace["steps"]):
            if require_labels and "mc_label" not in step:
                raise ValueError("unlabeled prefix. label all reasoning steps before training")
            row: Trace = {
                "text": prefix_text(trace["task"], trace["steps"][: index + 1]),
                "trace_id": trace["trace_id"],
                "step": index,
            }
            if "mc_label" in step:
                row["labels"] = [float(step["mc_label"]["probability"])]
            if "error_label" in step:
                row["error_label"] = step["error_label"]
            splits[trace["split"]].append(row)
    return splits


def first_error_metrics(
    traces: Sequence[Trace], probabilities: ArrayLike, threshold: float
) -> dict[str, float | int]:
    """Score exact first-error locations against independent step annotations.

    A wrong location contributes a false positive and a missed actual location.
    MC zeros are not substituted for independent error truth.
    """
    scores = np.asarray(probabilities, dtype=float)
    if not 0 <= threshold <= 1 or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("invalid scores or threshold. supply a probability vector")
    tp = fp = fn = exact = offset = 0
    for trace in traces:
        steps = trace["steps"]
        if any("error_label" not in step for step in steps):
            raise ValueError("independent error_label annotations are required for first-error F1")
        values = scores[offset : offset + len(steps)]
        if values.shape != (len(steps),) or np.any((values < 0) | (values > 1)):
            raise ValueError("one finite probability is required per step")
        actual = next((i for i, step in enumerate(steps) if step["error_label"]), None)
        predicted = next((i for i, value in enumerate(values) if value < threshold), None)
        match = actual == predicted
        exact += int(match)
        tp += int(predicted is not None and match)
        fp += int(predicted is not None and not match)
        fn += int(actual is not None and not match)
        offset += len(steps)
    if offset != len(scores) or not traces:
        raise ValueError("score count must match a nonempty trace collection")
    return {
        "first_error_precision": tp / max(tp + fp, 1),
        "first_error_recall": tp / max(tp + fn, 1),
        "first_error_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "first_error_exact_trace_accuracy": exact / len(traces),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
