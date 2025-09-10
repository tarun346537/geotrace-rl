"""Validated JSONL trace ingestion, synthetic fixtures and Monte Carlo labeling."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .constants import CUBE_SIZE_M, MC_SAMPLES, PARAMETERS, SEED, SUCCESS_TOLERANCE
from .physics import GravityTask, forward
from .process_rewards import exact_completion_probability, monte_carlo_label, prefixes_from_trace
from .survey_identity import SurveyRegistry
from .types import JsonRecord

FEATURE_NAMES = (
    "assigned_fraction",
    "x_assigned",
    "depth_assigned",
    "mass_assigned",
    "x_position",
    "depth_position",
    "mass_position",
    "log_survey_rms",
    "anomaly_center",
    "anomaly_spread",
    "default_model_relative_error",
)


def task_dict(task: GravityTask) -> JsonRecord:
    """Serialize a survey task without losing its backend or units."""
    return {
        "receivers_m": task.receivers_m.tolist(),
        "observations_microgal": task.observations_microgal.tolist(),
        "grid": {key: list(value) for key, value in task.grid.items()},
        "success_tolerance": task.success_tolerance,
        "forward_backend": task.forward_backend,
        "cube_size_m": task.cube_size_m,
    }


def parse_task(value: object) -> GravityTask:
    """Validate a serialized survey and construct its forward model."""
    required = {"receivers_m", "observations_microgal", "grid", "success_tolerance"}
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or not set(value) <= required | {"forward_backend", "cube_size_m"}
    ):
        raise ValueError("task must match the documented gravity survey schema")
    return GravityTask(
        np.asarray(value["receivers_m"]),
        np.asarray(value["observations_microgal"]),
        value["grid"],
        float(value["success_tolerance"]),
        value.get("forward_backend", "point_mass"),
        float(value.get("cube_size_m", 50.0)),
    )


def features(task: GravityTask, prefix: Mapping[str, float]) -> list[float]:
    """Observable survey/prefix features, without hidden truth or success labels."""
    task.validate_prefix(prefix)
    values = [len(prefix) / 3] + [float(key in prefix) for key in PARAMETERS]
    for key in PARAMETERS:
        grid = task.grid[key]
        value = prefix.get(key, grid[len(grid) // 2])
        values.append((value - min(grid)) / max(max(grid) - min(grid), 1e-12))
    y, x = task.observations_microgal, task.receivers_m
    weights = np.abs(y) / np.sum(np.abs(y))
    center = float(np.sum(weights * x))
    span = max(float(np.ptp(x)), 1.0)
    spread = float(np.sqrt(np.sum(weights * (x - center) ** 2)))
    model = {key: prefix.get(key, vals[len(vals) // 2]) for key, vals in task.grid.items()}
    default_error = np.sqrt(np.mean((task.prediction(model) - y) ** 2) / np.mean(y * y))
    values += [
        float(np.log1p(np.sqrt(np.mean(y * y)))),
        center / span,
        spread / span,
        float(np.log1p(default_error)),
    ]
    return values


def read_records(path: str | Path) -> list[JsonRecord]:
    """Read trace JSONL and reject identity or split leakage."""
    records: list[JsonRecord] = []
    surveys = SurveyRegistry()
    trace_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("every JSONL record must be an object")
                if not isinstance(record.get("survey_id"), str) or not record["survey_id"]:
                    raise ValueError("survey_id must be a nonempty string")
                if record.get("split") not in {"train", "validation", "test"}:
                    raise ValueError("split must be train, validation, or test")
                if not isinstance(record.get("trace_id"), str) or record["trace_id"] in trace_ids:
                    raise ValueError("trace_id must be unique")
                if not isinstance(record.get("provenance"), dict) or not record["provenance"].get(
                    "kind"
                ):
                    raise ValueError("explicit provenance.kind is required")
                task = parse_task(record["task"])
                surveys.add(record["survey_id"], record["split"], task)
                steps = record["assignments"]
                if (
                    not isinstance(steps, list)
                    or len(steps) != 3
                    or any(not isinstance(p, list) or len(p) != 2 for p in steps)
                ):
                    raise ValueError(
                        "a trace must contain exactly three [parameter,value] assignments"
                    )
                prefixes_from_trace(task, steps)
                trace_ids.add(record["trace_id"])
                records.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError("input JSONL contains no traces")
    return records


def write_jsonl(path: str | Path, records: Iterable[JsonRecord]) -> None:
    """Write finite JSON records, one complete trace per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8"
    )


def generate(
    surveys: int = 30,
    seed: int = SEED,
    backend: str = "point_mass",
    cube_size_m: float = CUBE_SIZE_M,
) -> list[JsonRecord]:
    """Create seeded synthetic survey worlds with disjoint survey splits."""
    if surveys < 10:
        raise ValueError("at least ten surveys are needed for grouped splits")
    rng = np.random.default_rng(seed)
    grid = {
        "x_m": (-100.0, -50.0, 0.0, 50.0, 100.0),
        "depth_m": (100.0, 150.0, 200.0, 250.0),
        "mass_kg": (6e9, 8e9, 1e10, 1.2e10, 1.4e10),
    }
    records = []
    for index in range(surveys):
        x = np.linspace(-450, 450, 25) + rng.uniform(-12, 12, 25)
        truth = {key: float(rng.choice(values)) for key, values in grid.items()}
        if backend == "simpeg_cube":
            from .simpeg_adapter import cube_forward

            clean = cube_forward(
                x,
                source_x_m=truth["x_m"],
                depth_m=truth["depth_m"],
                mass_kg=truth["mass_kg"],
                cube_size_m=cube_size_m,
            )
        elif backend == "point_mass":
            clean = forward(
                x, source_x_m=truth["x_m"], depth_m=truth["depth_m"], mass_kg=truth["mass_kg"]
            )
        else:
            raise ValueError("unknown gravity backend")
        y = clean + rng.normal(0, float(np.sqrt(np.mean(clean * clean))) * 0.003, x.size)
        task = GravityTask(x, y, grid, forward_backend=backend, cube_size_m=cube_size_m)
        split = "test" if index % 10 < 2 else "validation" if index % 10 < 4 else "train"
        for variant in range(4):
            model = dict(truth)
            if variant:
                key = PARAMETERS[variant - 1]
                model[key] = float(rng.choice([v for v in grid[key] if v != truth[key]]))
            assignments = [[key, model[key]] for key in PARAMETERS]
            records.append(
                {
                    "survey_id": f"synthetic-survey-{index:04d}",
                    "trace_id": f"synthetic-{index:04d}-{variant}",
                    "split": split,
                    "task": task_dict(task),
                    "assignments": assignments,
                    "provenance": {"kind": "synthetic_fixture", "seed": seed},
                }
            )
    return records


def import_survey(
    csv_path: str | Path,
    grid_path: str | Path,
    *,
    survey_id: str,
    split: str,
    backend: str = "point_mass",
    cube_size_m: float = CUBE_SIZE_M,
    tolerance: float = SUCCESS_TOLERANCE,
    trace_count: int = 8,
    seed: int = SEED,
    source_url: str | None = None,
) -> list[JsonRecord]:
    """Import supplied observations and create candidate traces, not labels."""
    if not survey_id or split not in {"train", "validation", "test"} or trace_count <= 0:
        raise ValueError("provide a survey ID, valid split, and positive trace_count")
    if source_url is not None and not source_url.startswith(("https://", "http://")):
        raise ValueError("source_url must identify the public data release")
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.array([float(row["x_m"]) for row in rows])
    y = np.array([float(row["gravity_microgal"]) for row in rows])
    grid = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    task = GravityTask(x, y, grid, tolerance, backend, cube_size_m)
    rng = np.random.default_rng(seed)
    source_digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    return [
        {
            "survey_id": survey_id,
            "trace_id": f"{survey_id}-candidate-{index}",
            "split": split,
            "task": task_dict(task),
            "assignments": [[key, float(rng.choice(task.grid[key]))] for key in PARAMETERS],
            "provenance": {
                "kind": "user_supplied_survey",
                "source_file": Path(csv_path).name,
                "source_sha256": source_digest,
                "source_url": source_url,
                "candidate_trace_generation_seed": seed,
            },
        }
        for index in range(trace_count)
    ]


def label_records(
    records: Iterable[JsonRecord], samples: int = MC_SAMPLES, seed: int = SEED
) -> list[JsonRecord]:
    """Estimate finite-grid process targets while retaining survey identity."""
    rng = np.random.default_rng(seed)
    output = []
    for record in records:
        task = parse_task(record["task"])
        prefixes = prefixes_from_trace(task, record["assignments"])
        rows = []
        for step, prefix in enumerate(prefixes):
            label = monte_carlo_label(task, prefix, samples=samples, rng=rng)
            # An enumeration oracle supplies evaluation truth only, never PRM input.
            rows.append(
                {
                    "step": step,
                    "features": features(task, prefix),
                    "mc_label": label.as_dict(),
                    "oracle_feasible_for_evaluation": exact_completion_probability(task, prefix)
                    > 0,
                }
            )
        output.append(
            {
                **record,
                "labeled_steps": rows,
                "labeling": {
                    "method": "uniform_grid_monte_carlo",
                    "samples": samples,
                    "seed": seed,
                },
            }
        )
    return output


def main() -> None:
    """Run the survey input and labeling commands."""
    from .data_cli import main as run

    run()


if __name__ == "__main__":
    main()
