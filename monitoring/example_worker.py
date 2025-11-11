"""Scalar-gradient worker for checkpoint and telemetry integration."""

import json
import os
import time

from geotrace.monitoring import WorkerContext
from geotrace.types import JsonRecord


def optimize_scalar(context: WorkerContext, config: JsonRecord) -> JsonRecord:
    """Run scalar gradient steps through the checkpoint and telemetry hooks."""
    restored = context.restore()
    if restored:
        committed_step, payload = restored
        state = json.loads(payload)
        if state["step"] != committed_step:
            raise ValueError("worker state and checkpoint manifest disagree")
    else:
        state = {"step": 0, "parameter": float(config.get("initial_parameter", 2.0))}
    target = float(config.get("target", 0.25))
    rate = float(config.get("learning_rate", 0.1))
    steps = int(config.get("steps", 5))
    if not 0 < rate < 1 or steps < state["step"]:
        raise ValueError("require 0<learning_rate<1 and target steps >= checkpoint progress")
    for step in range(state["step"] + 1, steps + 1):
        start = time.perf_counter()
        state["parameter"] -= rate * 2 * (state["parameter"] - target)
        state["step"] = step
        loss = (state["parameter"] - target) ** 2
        duration = max(time.perf_counter() - start, 1e-9)
        context.log_step(
            step,
            {"loss": loss, "parameter": state["parameter"]},
            elapsed_seconds=duration,
            allocated_gpus=0,
        )
        context.save_checkpoint(step, json.dumps(state, sort_keys=True).encode())
    return {"kind": "scalar_checkpoint_integration_example", **state, "target": target}


def exit_worker_once_after_checkpoint(context: WorkerContext, config: JsonRecord) -> JsonRecord:
    """Opt-in Ray failure test: kill this worker after its first durable step.

    Refuse to execute in the test driver itself. This intentionally exercises
    actual Ray worker-loss recovery. It is not used by the default run config.
    """
    if "driver_pid" not in config or os.getpid() == config["driver_pid"]:
        raise RuntimeError("fault injection is allowed only in a distinct Ray worker process")
    if context.attempt == 1:
        optimize_scalar(context, {**config, "steps": 1})
        os._exit(75)
    return optimize_scalar(context, config)
