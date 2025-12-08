"""Compare committed GPU work with externally measured allocation time."""

import math


def gpu_time_accounting(
    committed_gpu_hours: float, reserved_gpu_hours: float
) -> dict[str, float | str]:
    """Compute useful-time fraction with an explicit reservation denominator.

    Reserved time must include all allocated devices and the full experiment
    interval, including idle time and repeated work. It cannot be inferred from
    optimizer-step telemetry. Committed time counts retained completed steps.
    """
    if (
        isinstance(committed_gpu_hours, bool)
        or isinstance(reserved_gpu_hours, bool)
        or not math.isfinite(committed_gpu_hours)
        or not math.isfinite(reserved_gpu_hours)
        or committed_gpu_hours < 0
        or reserved_gpu_hours <= 0
        or reserved_gpu_hours < committed_gpu_hours
    ):
        raise ValueError(
            "require finite 0<=committed_gpu_hours<=reserved_gpu_hours and positive reserved time. "
            "check allocation timestamps, device counts and step durations"
        )
    return {
        "checkpoint_committed_gpu_hours": committed_gpu_hours,
        "reserved_gpu_hours": reserved_gpu_hours,
        "usable_gpu_hours_fraction": committed_gpu_hours / reserved_gpu_hours,
        "definition": "retained optimizer-step GPU hours / full measured reservation GPU hours",
    }
