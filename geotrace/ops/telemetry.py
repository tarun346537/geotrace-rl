"""Device measurements and optional experiment-tracker adapters."""

from __future__ import annotations

import csv
import io
import math
import subprocess
from collections.abc import Callable
from typing import Protocol

from ..types import JsonRecord
from .io import _utc


class WandbRun(Protocol):
    """The W&B SDK surface used by this adapter."""

    def log(self, data: JsonRecord) -> None:
        """Submit a measured metric record to the existing SDK run."""
        ...

    def alert(self, *, title: str, text: str, level: str, wait_duration: int) -> None:
        """Ask the SDK to submit a configured experiment alert."""
        ...


class TelemetrySink(Protocol):
    """An optional receiver for measured metrics and alert requests."""

    def log(self, record: JsonRecord) -> None:
        """Accept one measured optimizer-step record."""
        ...

    def alert(self, title: str, text: str, level: str) -> None:
        """Submit a named alert or raise a reportable sink error."""
        ...


class WandbSink:
    """Adapter around an already initialized W&B run, without creating an account/run."""

    def __init__(self, run: WandbRun) -> None:
        self.run = run

    def log(self, record: JsonRecord) -> None:
        """Submit measured metrics with explicit global-step and attempt axes."""
        # Do not reuse W&B's internal step after recovery. Retain explicit axes.
        self.run.log(
            {**record["metrics"], "global_step": record["step"], "attempt": record["attempt"]}
        )

    def alert(self, title: str, text: str, level: str) -> None:
        """Submit an alert through the initialized W&B run."""
        from wandb.sdk.wandb_alerts import AlertLevel

        self.run.alert(title=title, text=text, level=getattr(AlertLevel, level), wait_duration=0)


def sample_nvidia_smi(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> JsonRecord:
    """Return measured GPU utilization/memory, or an explicit unavailable reason."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=5, check=True)
        devices = []
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != 5:
                raise ValueError("unexpected nvidia-smi output shape")
            index, uuid, utilization, used, total = [value.strip() for value in row]
            device: JsonRecord = {
                "index": int(index),
                "uuid": uuid,
                "utilization_percent": float(utilization),
                "memory_used_mib": float(used),
                "memory_total_mib": float(total),
            }
            if (
                not uuid
                or not all(
                    math.isfinite(device[key])
                    for key in ("utilization_percent", "memory_used_mib", "memory_total_mib")
                )
                or not 0 <= device["utilization_percent"] <= 100
                or not 0 <= device["memory_used_mib"] <= device["memory_total_mib"]
            ):
                raise ValueError("invalid GPU utilization or memory sample")
            devices.append(device)
        return {"available": bool(devices), "devices": devices, "sampled_at_utc": _utc()}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "available": False,
            "devices": [],
            "error": f"{type(exc).__name__}: {exc}",
            "sampled_at_utc": _utc(),
        }
