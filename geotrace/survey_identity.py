"""Dataset identity checks shared by assignment and textual traces."""

from __future__ import annotations

import hashlib
import json

from .physics import GravityTask


class SurveyRegistry:
    """Keep a survey's observations, task definition and split fixed."""

    def __init__(self) -> None:
        self.surveys: dict[str, tuple[str, str]] = {}
        self.observations: dict[str, str] = {}

    def add(self, survey_id: str, split: str, task: GravityTask) -> None:
        """Reject renamed observation sets crossing splits and changed tasks."""
        # Sort pairs together so reversing a CSV cannot bypass the split check.
        pairs = sorted(
            zip(task.receivers_m.tolist(), task.observations_microgal.tolist(), strict=True)
        )
        observed = json.dumps(pairs, separators=(",", ":"), allow_nan=False)
        observation_hash = hashlib.sha256(observed.encode()).hexdigest()
        definition = json.dumps(
            {
                "observations": pairs,
                "grid": {key: list(values) for key, values in task.grid.items()},
                "success_tolerance": task.success_tolerance,
                "forward_backend": task.forward_backend,
                "cube_size_m": task.cube_size_m,
            },
            sort_keys=True,
            allow_nan=False,
        )
        task_hash = hashlib.sha256(definition.encode()).hexdigest()
        prior = self.surveys.setdefault(survey_id, (split, task_hash))
        if prior[0] != split:
            raise ValueError("a survey cannot appear in multiple splits. assign it to one split")
        if prior[1] != task_hash:
            raise ValueError("survey_id has conflicting tasks. keep its survey and solver fixed")
        if self.observations.setdefault(observation_hash, split) != split:
            raise ValueError("identical survey observations cross splits under different IDs")
