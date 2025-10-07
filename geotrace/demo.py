"""Seeded analytic gravity simulation and GRPO objective calculation."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .grpo import clipped_objective, group_relative_advantages
from .logging_config import configure_logging
from .physics import synthetic_task
from .process_rewards import (
    bounded_reward,
    exact_completion_probability,
    first_irrecoverable_step,
    monte_carlo_label,
    prefixes_from_trace,
    verify_tool_output,
)
from .types import JsonRecord

LOGGER = logging.getLogger(__name__)


def run(seed: int = 17, samples: int = 512) -> JsonRecord:
    """Compute deterministic gravity-prefix labels and objective diagnostics."""
    rng = np.random.default_rng(seed)
    task, truth = synthetic_task()
    traces = {
        "consistent": [("x_m", 0.0), ("depth_m", 150.0), ("mass_kg", 1e10)],
        "wrong_depth": [("x_m", 0.0), ("depth_m", 250.0), ("mass_kg", 1e10)],
        "wrong_position": [("x_m", 100.0), ("depth_m", 150.0), ("mass_kg", 1e10)],
    }
    reports, rewards = {}, []
    for name, assignments in traces.items():
        prefixes = prefixes_from_trace(task, assignments)
        labels = [monte_carlo_label(task, prefix, samples=samples, rng=rng) for prefix in prefixes]
        final_prediction = task.prediction(prefixes[-1])
        valid = verify_tool_output(task, prefixes[-1], final_prediction)
        reward = bounded_reward(
            float(np.mean([label.probability for label in labels])), physics_valid=valid
        )
        rewards.append(reward)
        reports[name] = {
            "assignments": assignments,
            "process_labels": [label.as_dict() for label in labels],
            "exact_probabilities_for_demo_evaluation": [
                exact_completion_probability(task, p) for p in prefixes
            ],
            "oracle_first_irrecoverable_step_zero_based": first_irrecoverable_step(
                task, assignments
            ),
            "physics_valid": valid,
            "mean_process_reward": reward,
        }
    expected = task.prediction(truth)
    forged = np.zeros_like(expected)  # Looks innocuous but contradicts the stated forward call.
    forged_valid = verify_tool_output(task, truth, forged)
    forged_reward = bounded_reward(0.99, physics_valid=forged_valid)
    rewards.append(forged_reward)
    advantages = group_relative_advantages([rewards])[0]
    # Fixed token probabilities make the objective calculation reproducible.
    old_logp = np.full((4, 3), np.log(0.2))
    logp = old_logp + np.array([[0.08], [-0.02], [-0.05], [-0.10]])
    mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]])
    objective = clipped_objective(logp, old_logp, old_logp, advantages, mask)
    return {
        "provenance": {
            "kind": "synthetic_demo",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
        },
        "workload": "analytic gravity simulation, Monte Carlo labels, verifier and GRPO objective arithmetic",
        "task": {
            "receivers": len(task.receivers_m),
            "grid_models": 100,
            "success_relative_rmse_threshold": task.success_tolerance,
            "synthetic_truth": truth,
        },
        "traces": reports,
        "forged_tool_response": {
            "physics_valid": forged_valid,
            "unverified_score": 0.99,
            "gated_reward": forged_reward,
        },
        "group_relative_advantages": advantages.tolist(),
        "grpo_objective": objective,
    }


def main() -> None:
    """Write the synthetic gravity and process-reward report."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo.json"))
    args = parser.parse_args()
    if args.samples <= 0 or args.seed < 0:
        parser.error("samples must be positive and seed must be nonnegative")
    report = run(args.seed, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("gravity_demo_written", extra={"output": str(args.output)})


if __name__ == "__main__":
    main()
