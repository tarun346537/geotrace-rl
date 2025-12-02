"""Ray Jobs entry point for a monitored native verl training attempt."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from .ops.context import WorkerContext
from .ops.io import _atomic, _json
from .ops.verl_checkpoint import bind_native_directory, restore_native
from .types import JsonRecord


def infrastructure_failure(error: BaseException, exceptions: ModuleType) -> bool:
    """Retry confirmed Ray worker/node loss, never arbitrary task exceptions."""
    seen = set()
    while isinstance(error, exceptions.RayTaskError) and id(error) not in seen:
        seen.add(id(error))
        error = error.cause
    return isinstance(
        error,
        (exceptions.WorkerCrashedError, exceptions.NodeDiedError, exceptions.LocalRayletDiedError),
    ) or (isinstance(error, exceptions.RayActorError) and error.preempted)


def run_attempt(
    context: WorkerContext,
    run: Callable[[], None],
    classify: Callable[[BaseException], bool],
) -> None:
    """Write a trusted terminal record for the external job supervisor."""
    attempt_dir = Path(context.directory) / f"attempt-{context.attempt}"
    try:
        run()
        checkpoint = restore_native(context)
        if checkpoint is None:
            raise RuntimeError("native training finished without a committed checkpoint")
        result = {"run_id": context.run_id, "attempt": context.attempt, "checkpoint": checkpoint}
        _atomic(attempt_dir / "result.json", (_json(result) + "\n").encode())
    except Exception as exc:
        failure = {
            "run_id": context.run_id,
            "attempt": context.attempt,
            "retryable": bool(classify(exc)),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic(attempt_dir / "failure.json", (_json(failure) + "\n").encode())
        raise


def configure_resume(configuration: object, context: WorkerContext) -> None:
    """Force native restoration to the last ledger-committed global step."""
    config = cast(Any, configuration)
    root = Path(config.trainer.default_local_dir).resolve()
    bind_native_directory(root, context)
    config.trainer.default_local_dir = str(root)
    saved = restore_native(context)
    if saved and Path(saved["path"]).parent != root:
        raise ValueError("native checkpoint root changed. retain the original run directory")
    config.trainer.resume_mode = "resume_path" if saved else "disable"
    config.trainer.resume_from_path = saved["path"] if saved else None
    if config.trainer.save_freq <= 0 or config.trainer.del_local_ckpt_after_load:
        raise ValueError("native recovery requires save_freq>0 and del_local_ckpt_after_load=False")
    for role in (config.actor_rollout_ref.actor, config.critic):
        if role.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(
                "native recovery supports FSDP checkpoints. select the pinned launchers"
            )
        if not {"model", "optimizer", "extra"} <= set(role.checkpoint.save_contents):
            raise ValueError("native recovery needs model, optimizer and extra checkpoint contents")


def main() -> None:
    """Compose native configuration, attach Ray hooks and run the real trainer."""
    if importlib.metadata.version("verl") != "0.5.0":
        raise RuntimeError("use verl==0.5.0 for the native training connector")
    spec: JsonRecord = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    context = WorkerContext(**spec["context"])

    def _train() -> None:
        import hydra
        import ray
        import verl

        module = importlib.import_module("verl.trainer.main_ppo")
        constants = importlib.import_module("verl.trainer.constants_ppo")
        with hydra.initialize_config_dir(
            config_dir=str(Path(verl.__file__).parent / "trainer/config"), version_base=None
        ):
            config = hydra.compose(config_name="ppo_trainer", overrides=spec["overrides"])
        configure_resume(config, context)
        gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
        environment = dict(spec["environment"])
        environment.update(
            GEOTRACE_RUN_ID=context.run_id,
            GEOTRACE_ATTEMPT=str(context.attempt),
            GEOTRACE_RUN_DIRECTORY=context.directory,
            GEOTRACE_ALLOCATED_GPUS=str(gpus),
        )
        os.environ.update(environment)
        runtime = constants.get_ppo_ray_runtime_env()
        runtime.setdefault("env_vars", {}).update(environment)
        runtime["worker_process_setup_hook"] = "geotrace.verl_hooks.install"
        ray.init(address="auto", runtime_env=runtime)
        try:
            module.run_ppo(config)
        finally:
            ray.shutdown()

    exceptions = importlib.import_module("ray.exceptions")
    run_attempt(context, _train, lambda error: infrastructure_failure(error, exceptions))


if __name__ == "__main__":
    main()
