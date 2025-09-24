"""Survey identities retained with reward-model exports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

from .data import parse_task
from .survey_identity import SurveyRegistry


class Provenance(TypedDict):
    """Split assignments for both declared IDs and physical observations."""

    schema_version: int
    observations: dict[str, str]
    survey_ids: dict[str, str]


def survey_provenance(records: Sequence[Mapping[str, Any]]) -> Provenance:
    """Fingerprint observations without retaining labels or raw survey values."""
    registry = SurveyRegistry()
    for record in records:
        name, split = record["survey_id"], record["split"]
        if not isinstance(name, str) or not name or split not in {"train", "validation", "test"}:
            raise ValueError("survey provenance needs valid IDs and split assignments")
        registry.add(name, split, parse_task(record["task"]))
    if not registry.surveys:
        raise ValueError("survey provenance cannot be empty. supply recorded tasks")
    return {
        "schema_version": 1,
        "observations": dict(registry.observations),
        "survey_ids": {key: split for key, (split, _) in registry.surveys.items()},
    }


def read_provenance(path: str | Path) -> Provenance:
    """Read assignment or textual JSONL records into the same identity contract."""
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in records):
        raise ValueError("survey records must be JSON objects. check the input JSONL")
    return survey_provenance(records)


def _checked(value: object) -> Provenance:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("checkpoint lacks survey provenance. export it with the current trainer")
    for field in ("observations", "survey_ids"):
        rows = value.get(field)
        if (
            not isinstance(rows, dict)
            or not rows
            or any(
                not isinstance(key, str) or split not in {"train", "validation", "test"}
                for key, split in rows.items()
            )
        ):
            raise ValueError("invalid checkpoint survey provenance. export the trained head again")
    return cast(Provenance, value)


def validate_heldout(current: Provenance, saved: object, split: str = "test") -> None:
    """Reject renamed training/calibration observations in held-out evaluation."""
    previous = _checked(saved)
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation requires a validation or test split")
    found = False
    for field in ("observations", "survey_ids"):
        for identity, role in current[field].items():
            if role != split:
                continue
            found = True
            prior = previous[field].get(identity)
            if prior is not None and prior != split:
                raise ValueError(
                    "evaluation overlaps another checkpoint split. supply independent held-out surveys"
                )
    if not found:
        raise ValueError(f"no {split} surveys found. check evaluation split assignments")
