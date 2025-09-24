"""Feature-head data, evaluation, and atomic model exports."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn

from .data import FEATURE_NAMES, read_records
from .prm_settings import defaults

Record = dict[str, Any]
Metrics = dict[str, float | int]


class FeatureSplit(TypedDict):
    """Arrays from one survey-disjoint feature dataset split."""

    x: NDArray[np.float32]
    y: NDArray[np.float32]
    feasible: NDArray[np.bool_]
    trace_id: list[str]


class ProcessRewardHead(nn.Module):
    """Small feature baseline for Monte Carlo continuation probabilities."""

    def __init__(self, hidden: int | None = None) -> None:
        super().__init__()
        width = int(defaults("feature")["hidden"]) if hidden is None else hidden
        if width < 1:
            raise ValueError("hidden width must be positive. check feature settings")
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one logit for each observable feature vector."""
        return cast(torch.Tensor, self.network(features).squeeze(-1))


def load_dataset(path: str | Path) -> dict[str, FeatureSplit]:
    """Validate labeled prefixes and collect independent survey splits."""
    records = read_records(path)
    splits: dict[str, FeatureSplit] = {}
    for name in ("train", "validation", "test"):
        xs: list[NDArray[np.float32]] = []
        ys: list[float] = []
        feasible_values: list[bool] = []
        ids: list[str] = []
        for record in records:
            if record["split"] != name:
                continue
            steps = record.get("labeled_steps")
            if not isinstance(steps, list) or len(steps) != 3:
                raise ValueError("missing labeled steps. run geotrace.data label first")
            for index, row in enumerate(steps):
                x = np.asarray(row["features"], dtype=np.float32)
                y = float(row["mc_label"]["probability"])
                feasible = row["oracle_feasible_for_evaluation"]
                if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all() or not 0 <= y <= 1:
                    raise ValueError("invalid features or MC target. relabel the input traces")
                if row["step"] != index or not isinstance(feasible, bool):
                    raise ValueError("invalid step order or annotation. check labeled traces")
                xs.append(x)
                ys.append(y)
                feasible_values.append(feasible)
                ids.append(str(record["trace_id"]))
        if not xs:
            raise ValueError(f"empty {name} split. provide independent surveys for every split")
        splits[name] = {
            "x": np.asarray(xs, dtype=np.float32),
            "y": np.asarray(ys, dtype=np.float32),
            "feasible": np.asarray(feasible_values, dtype=bool),
            "trace_id": ids,
        }
    return splits


def f1_error(probability: ArrayLike, feasible: ArrayLike, threshold: float) -> float:
    """Calculate error-positive F1 at a fixed feasibility threshold."""
    scores, valid = np.asarray(probability, float), np.asarray(feasible, bool)
    if scores.shape != valid.shape or not 0 <= threshold <= 1 or not np.isfinite(scores).all():
        raise ValueError(
            "invalid F1 inputs. use matched scores, annotations and a probability threshold"
        )
    predicted, actual = scores < threshold, ~valid
    tp = int(np.sum(predicted & actual))
    fp, fn = int(np.sum(predicted & ~actual)), int(np.sum(~predicted & actual))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def evaluate(
    model: ProcessRewardHead, split: FeatureSplit, mean: ArrayLike, std: ArrayLike, threshold: float
) -> tuple[Metrics, NDArray[np.float64]]:
    """Evaluate a head without changing training normalization or its threshold."""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor((split["x"] - np.asarray(mean)) / np.asarray(std))
        probability = np.asarray(torch.sigmoid(model(inputs)).cpu().numpy(), dtype=float)
    groups: dict[str, list[int]] = {}
    for index, trace in enumerate(split["trace_id"]):
        groups.setdefault(trace, []).append(index)
    matched = 0
    for indices in groups.values():
        actual = next((i for i, row in enumerate(indices) if not split["feasible"][row]), None)
        predicted = next((i for i, row in enumerate(indices) if probability[row] < threshold), None)
        matched += int(actual == predicted)
    metrics: Metrics = {
        "soft_target_mse": float(np.mean((probability - split["y"]) ** 2)),
        "step_error_f1": f1_error(probability, split["feasible"], threshold),
        "first_error_exact_trace_accuracy": matched / len(groups),
        "steps": len(probability),
        "traces": len(groups),
        "threshold": threshold,
    }
    return metrics, probability


def atomic_save(path: str | Path, state: Mapping[str, Any]) -> None:
    """Replace an export only after writing the complete temporary payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, target)


def load_head(path: str | Path) -> tuple[ProcessRewardHead, Record]:
    """Load a local restricted state dictionary for feature-head inference."""
    state = cast(Record, torch.load(path, map_location="cpu", weights_only=True))
    model = ProcessRewardHead(int(state["hidden"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, state
