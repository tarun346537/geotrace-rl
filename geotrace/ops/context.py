"""Measured-step and checkpoint hooks provided to workers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..types import JsonRecord
from .checkpoint import CheckpointStore
from .io import _append, _utc


@dataclass
class WorkerContext:
    """Per-attempt checkpoint and measured-step training hooks."""

    run_id: str
    attempt: int
    directory: str
    observer: Callable[[], None] | None = None

    @property
    def store(self) -> CheckpointStore:
        """Return the logical run's shared checkpoint store."""
        return CheckpointStore(Path(self.directory) / "checkpoints")

    def restore(self) -> tuple[int, bytes] | None:
        """Return (last committed global step, bytes), or None on a fresh run."""
        return self.store.load()

    def save_checkpoint(self, step: int, payload: bytes) -> JsonRecord:
        """Commit state at a completed global step."""
        manifest = self.store.save(step=step, attempt=self.attempt, payload=payload)
        if self.observer:
            self.observer()
        return manifest

    def log_step(
        self,
        step: int,
        metrics: Mapping[str, float],
        *,
        elapsed_seconds: float,
        allocated_gpus: float = 0,
    ) -> None:
        """Log one completed optimizer step, including reasoning-step summaries.

        Example metrics: process_reward_mean, process_reward_min,
        physics_failure_fraction, kl_mean, loss, gpu_utilization_percent.
        elapsed_seconds is measured step duration, never an assumed cadence.
        """
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if (
            not math.isfinite(elapsed_seconds)
            or elapsed_seconds <= 0
            or not math.isfinite(allocated_gpus)
            or allocated_gpus < 0
        ):
            raise ValueError("step duration must be positive and GPU count nonnegative")
        if (
            not isinstance(metrics, Mapping)
            or not metrics
            or any(not isinstance(key, str) for key in metrics)
        ):
            raise ValueError("metrics must be a nonempty mapping of names to finite scalar values")
        bad = [
            key
            for key, value in metrics.items()
            if not isinstance(value, (int, float)) or not math.isfinite(value)
        ]
        if bad:
            _append(
                Path(self.directory) / f"attempt-{self.attempt}" / "events.jsonl",
                {"kind": "nonfinite_metric", "step": step, "names": bad, "at_utc": _utc()},
            )
            raise ValueError(f"nonfinite or nonscalar step metrics: {bad}")
        for name in ("physics_failure_fraction",):
            if name in metrics and not 0 <= metrics[name] <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if (
            "gpu_utilization_percent" in metrics
            and not 0 <= metrics["gpu_utilization_percent"] <= 100
        ):
            raise ValueError("GPU utilization must be in [0,100]")
        row = {
            "run_id": self.run_id,
            "attempt": self.attempt,
            "step": step,
            "metrics": dict(metrics),
            "elapsed_seconds": float(elapsed_seconds),
            "allocated_gpus": float(allocated_gpus),
            "at_utc": _utc(),
        }
        _append(Path(self.directory) / f"attempt-{self.attempt}" / "metrics.jsonl", row)
        if self.observer:
            self.observer()
