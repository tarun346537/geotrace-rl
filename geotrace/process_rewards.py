"""Monte Carlo prefix labels and finite-grid first-error evaluation."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from .constants import WILSON_Z
from .physics import GravityTask


@dataclass(frozen=True)
class ProcessLabel:
    """Sampled success probability and its Wilson interval for one prefix."""

    successes: int
    samples: int
    probability: float
    wilson_lower: float
    wilson_upper: float

    def as_dict(self) -> dict[str, int | float]:
        """Expose serializable label statistics without changing their values."""
        return asdict(self)


def monte_carlo_label(
    task: GravityTask, prefix: Mapping[str, float], *, samples: int, rng: np.random.Generator
) -> ProcessLabel:
    """Estimate probability of final success under uniform grid completion.

    A prefix is a set of irrevocable parameter assignments. Unassigned
    parameters are sampled independently from the grid WITH replacement.
    The result is specific to that continuation policy.
    """
    task.validate_prefix(prefix)
    if not isinstance(samples, (int, np.integer)) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    successes = 0
    for _ in range(samples):
        model = {
            key: prefix[key] if key in prefix else float(rng.choice(values))
            for key, values in task.grid.items()
        }
        successes += int(task.successful(model))
    probability = successes / samples
    z = WILSON_Z
    denominator = 1 + z * z / samples
    center = (probability + z * z / (2 * samples)) / denominator
    half = (
        z
        * np.sqrt(probability * (1 - probability) / samples + z * z / (4 * samples**2))
        / denominator
    )
    return ProcessLabel(
        successes,
        int(samples),
        probability,
        float(max(0, center - half)),
        float(min(1, center + half)),
    )


def exact_completion_probability(task: GravityTask, prefix: Mapping[str, float]) -> float:
    """Evaluate the finite-grid continuation probability by enumeration."""
    outcomes = [task.successful(model) for model in task.completions(prefix)]
    return float(np.mean(outcomes))


def prefixes_from_trace(
    task: GravityTask, assignments: Sequence[tuple[str, float]]
) -> list[dict[str, float]]:
    """Validate an append-only assignment trace and materialize its prefixes."""
    prefix: dict[str, float] = {}
    prefixes = []
    for key, value in assignments:
        if key in prefix:
            raise ValueError("trace assignments are irrevocable. a key may appear once")
        prefix = {**prefix, key: value}
        task.validate_prefix(prefix)
        prefixes.append(prefix)
    return prefixes


def first_irrecoverable_step(
    task: GravityTask, assignments: Sequence[tuple[str, float]]
) -> int | None:
    """Zero-based first prefix with no successful completion, by enumeration.

    The finite-grid oracle evaluates every parameter completion at each prefix.
    """
    for index, prefix in enumerate(prefixes_from_trace(task, assignments)):
        if not any(task.successful(model) for model in task.completions(prefix)):
            return index
    return None


def verify_tool_output(
    task: GravityTask,
    model: Mapping[str, float],
    claimed_prediction: ArrayLike,
    *,
    relative_tolerance: float = 1e-8,
) -> bool:
    """Recompute a forward call independently of the claimed tool response."""
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("verification tolerance must be finite and nonnegative")
    expected = task.prediction(model)
    claimed = np.asarray(claimed_prediction, dtype=float)
    if claimed.shape != expected.shape or not np.all(np.isfinite(claimed)):
        return False
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    return bool(np.max(np.abs(claimed - expected)) <= relative_tolerance * scale)


def bounded_reward(
    process_score: float, *, physics_valid: bool, lower: float = -1.0, upper: float = 1.0
) -> float:
    """Gate inconsistent tool calls before clipping a finite process score."""
    if not np.all(np.isfinite([process_score, lower, upper])) or lower >= upper:
        raise ValueError("reward and ordered clipping bounds must be finite")
    return float(np.clip(process_score if physics_valid else lower, lower, upper))
