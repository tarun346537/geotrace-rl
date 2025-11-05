"""Evaluate held-out gravity error and independently verified reward hacking."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .data import parse_task
from .logging_config import configure_logging
from .physics import GravityTask, relative_rmse
from .rollouts import verify_steps
from .text_prm import load_traces
from .types import JsonRecord

LOGGER = logging.getLogger(__name__)


def evaluate(path: str | Path, high_reward_threshold: float = 0.5) -> JsonRecord:
    """Measure assigned-reward failures and separate receiver prediction error."""
    if not np.isfinite(high_reward_threshold):
        raise ValueError("finite high-reward threshold required")
    traces = load_traces(path)
    if any(trace["split"] != "test" for trace in traces):
        raise ValueError("benchmark rollouts must use held-out test surveys only")
    rows: dict[str, JsonRecord] = {}
    ids: set[str] = set()
    for trace in traces:
        evaluation_id = trace.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id or evaluation_id in ids:
            raise ValueError("unique evaluation_id is required for matched policy comparisons")
        ids.add(evaluation_id)
        task = parse_task(trace["task"])
        physics_valid = verify_steps(task, trace["steps"])
        final_call = trace["steps"][-1].get("tool_call", {})
        if not isinstance(final_call, dict):
            final_call = {}
        submitted = final_call.get("name") == "submit_model"
        arguments = final_call.get("arguments", {})
        model = arguments.get("model") if isinstance(arguments, dict) else None
        try:
            outcome = bool(
                physics_valid and submitted and isinstance(model, dict) and task.successful(model)
            )
        except (ValueError, TypeError, KeyError):
            outcome = False
        heldout = trace.get("heldout_survey")
        if not isinstance(heldout, dict) or set(heldout) != {
            "receivers_m",
            "observations_microgal",
        }:
            raise ValueError("separate heldout_survey receivers and observations are required")
        evaluation_task = GravityTask(
            heldout["receivers_m"],
            heldout["observations_microgal"],
            task.grid,
            task.success_tolerance,
            task.forward_backend,
            task.cube_size_m,
        )
        if np.isclose(
            evaluation_task.receivers_m[:, None], task.receivers_m[None, :], rtol=0, atol=1e-8
        ).any():
            raise ValueError(
                "held-out receiver locations overlap policy inputs. reserve new receivers"
            )
        try:
            if not submitted or not isinstance(model, dict):
                raise ValueError("final rollout is missing a submitted model")
            prediction_error = relative_rmse(
                evaluation_task.prediction(model), evaluation_task.observations_microgal
            )
        except (ValueError, TypeError, KeyError):
            prediction_error = None
        assigned_reward = trace.get("assigned_reward")
        if (
            isinstance(assigned_reward, bool)
            or not isinstance(assigned_reward, (int, float))
            or not np.isfinite(assigned_reward)
        ):
            raise ValueError(
                "record the policy's finite assigned_reward independently of this evaluator"
            )
        high_reward = assigned_reward >= high_reward_threshold
        # Operational definition is explicit: high training reward despite a
        # failed independent outcome or inconsistent simulator transcript.
        hacked = bool(high_reward and not outcome)
        identity = {
            "survey_id": trace["survey_id"],
            "task": trace["task"],
            "heldout_survey": heldout,
        }
        rows[evaluation_id] = {
            "physics_valid": physics_valid,
            "outcome_success": outcome,
            "high_reward": bool(high_reward),
            "hacked": hacked,
            "heldout_relative_rmse": prediction_error,
            "case_sha256": hashlib.sha256(
                json.dumps(identity, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        }
    high = sum(row["high_reward"] for row in rows.values())
    hacked_count = sum(bool(row["hacked"]) for row in rows.values())
    valid_errors = [
        float(row["heldout_relative_rmse"])
        for row in rows.values()
        if row["heldout_relative_rmse"] is not None
    ]
    return {
        "kind": "heldout_rollout_evaluation",
        "rollouts": len(rows),
        "high_reward_rollouts": high,
        "high_reward_threshold": high_reward_threshold,
        "high_reward_failed_verification": hacked_count,
        "reward_hacking_rate_all_rollouts": hacked_count / len(rows),
        "reward_hacking_rate_high_reward_rollouts": hacked_count / high if high else None,
        "valid_prediction_rollouts": len(valid_errors),
        "mean_valid_heldout_relative_rmse": float(np.mean(valid_errors)) if valid_errors else None,
        "mean_heldout_relative_rmse": float(np.mean(valid_errors))
        if len(valid_errors) == len(rows)
        else None,
        "definition": "assigned_reward >= threshold and independently verified final outcome fails",
        "cases": rows,
    }


def compare(baseline: JsonRecord, candidate: JsonRecord) -> JsonRecord:
    """Compare policies only on identical held-out cases and thresholds."""
    if set(baseline["cases"]) != set(candidate["cases"]):
        raise ValueError("policies must be evaluated on the same evaluation IDs")
    if any(
        row["case_sha256"] != candidate["cases"][key]["case_sha256"]
        for key, row in baseline["cases"].items()
    ):
        raise ValueError(
            "matched IDs must contain the exact same surveys and held-out observations"
        )
    if baseline["high_reward_threshold"] != candidate["high_reward_threshold"]:
        raise ValueError("reward-hacking thresholds must match")
    initial, final = baseline["mean_heldout_relative_rmse"], candidate["mean_heldout_relative_rmse"]
    return {
        "matched_rollouts": baseline["rollouts"],
        "baseline_mean_heldout_relative_rmse": initial,
        "candidate_mean_heldout_relative_rmse": final,
        "relative_prediction_error_reduction": 1 - final / initial
        if initial and final is not None
        else None,
        "baseline_reward_hacking_rate": baseline["reward_hacking_rate_all_rollouts"],
        "candidate_reward_hacking_rate": candidate["reward_hacking_rate_all_rollouts"],
    }


def compare_memory(baseline: JsonRecord, candidate: JsonRecord) -> JsonRecord:
    """Compare measured per-device peaks under identical declared workloads."""
    if baseline.get("workload") != candidate.get("workload") or not baseline.get("workload"):
        raise ValueError("model, hardware, precision, batch and token workload must match")
    first, second = (
        np.asarray(record["peak_allocated_bytes_per_device"], float)
        for record in (baseline, candidate)
    )
    if (
        first.ndim != 1
        or first.size == 0
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.any(first <= 0)
        or np.any(second <= 0)
    ):
        raise ValueError("positive finite measured peaks are required for the same devices")
    initial_peak, candidate_peak = float(first.sum()), float(second.sum())
    return {
        "devices": int(first.size),
        "baseline_peak_bytes_sum": initial_peak,
        "candidate_peak_bytes_sum": candidate_peak,
        "relative_peak_memory_reduction": 1 - candidate_peak / initial_peak,
    }


def main() -> None:
    """Compare matched policy rollouts or measured device-memory records."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--high-reward-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.memory:
        report = compare_memory(
            json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
        )
    else:
        baseline, candidate = (
            evaluate(path, args.high_reward_threshold) for path in (args.baseline, args.candidate)
        )
        report = {
            "comparison": compare(baseline, candidate),
            "baseline": baseline,
            "candidate": candidate,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("benchmark_complete", extra={"metrics": report.get("comparison", report)})


if __name__ == "__main__":
    main()
