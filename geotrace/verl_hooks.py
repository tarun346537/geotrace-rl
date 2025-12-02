"""Pinned verl 0.5 hooks installed by Ray's worker setup callback."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Mapping
from functools import wraps
from numbers import Real
from pathlib import Path
from typing import Any, cast

import numpy as np

from .ops.context import WorkerContext
from .ops.io import _append
from .ops.verl_checkpoint import publish_native


def reward_metrics(batch: object) -> dict[str, float]:
    """Summarize actual prefix scores and independent physics failures per update."""
    values = cast(Any, batch).non_tensor_batch
    metrics = {}
    if "physics_valid" in values:
        metrics["physics_failure_fraction"] = 1 - float(np.mean(values["physics_valid"]))
    if "process_count" in values:
        counts = np.asarray(values["process_count"], dtype=float)
        if counts.sum() > 0:
            metrics["process_reward_mean"] = float(np.sum(values["process_sum"]) / counts.sum())
            metrics["process_reward_min"] = float(
                np.min(np.asarray(values["process_min"])[counts > 0])
            )
            metrics["reasoning_steps_scored"] = float(counts.sum())
    return metrics


def install_hooks(
    tracking_type: object, module: object, context: WorkerContext, allocated_gpus: int
) -> None:
    """Attach measured native training events without changing optimizer behavior."""
    tracking, trainer_module = cast(Any, tracking_type), cast(Any, module)
    if getattr(tracking, "_geotrace_hooks", False):
        return
    original_log = tracking.log
    original_metrics = trainer_module.compute_data_metrics
    original_save = trainer_module.RayPPOTrainer._save_checkpoint

    @wraps(original_metrics)
    def _metrics(batch: object, use_critic: bool = True) -> dict[str, Any]:
        return {**original_metrics(batch, use_critic=use_critic), **reward_metrics(batch)}

    @wraps(original_log)
    def _log(
        self: object, data: Mapping[str, Any], step: int, backend: str | list[str] | None = None
    ) -> None:
        scalars = {key: float(value) for key, value in data.items() if isinstance(value, Real)}
        if "training/global_step" in scalars:
            if int(scalars["training/global_step"]) != step:
                raise ValueError("native telemetry step differs from optimizer step")
            if "actor/kl_loss" in scalars:
                scalars["kl_mean"] = scalars["actor/kl_loss"]
            context.log_step(
                step,
                scalars,
                elapsed_seconds=scalars["timing_s/step"],
                allocated_gpus=allocated_gpus,
            )
        else:
            _append(
                Path(context.directory) / f"attempt-{context.attempt}" / "validation.jsonl",
                {
                    "run_id": context.run_id,
                    "attempt": context.attempt,
                    "step": step,
                    "metrics": scalars,
                },
            )
        original_log(self, data=data, step=step, backend=backend)

    @wraps(original_save)
    def _save(instance: object) -> None:
        self = cast(Any, instance)
        original_save(self)
        publish_native(
            context,
            Path(self.config.trainer.default_local_dir),
            self.global_steps,
            allocated_gpus,
            self.use_critic,
        )

    tracking.log = _log
    trainer_module.compute_data_metrics = _metrics
    trainer_module.RayPPOTrainer._save_checkpoint = _save
    tracking._geotrace_hooks = True


def install() -> None:
    """Install callbacks once in a Ray worker using its supplied run context."""
    if importlib.metadata.version("verl") != "0.5.0":
        raise RuntimeError("native hooks require verl==0.5.0. use the pinned CUDA environment")
    context = WorkerContext(
        os.environ["GEOTRACE_RUN_ID"],
        int(os.environ["GEOTRACE_ATTEMPT"]),
        os.environ["GEOTRACE_RUN_DIRECTORY"],
    )
    install_hooks(
        importlib.import_module("verl.utils.tracking").Tracking,
        importlib.import_module("verl.trainer.ppo.ray_trainer"),
        context,
        int(os.environ["GEOTRACE_ALLOCATED_GPUS"]),
    )
