"""Local and Ray worker execution with explicit infrastructure errors."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from typing import Protocol

from ..types import JsonRecord
from .context import WorkerContext


class Backend(Protocol):
    """Run one attempt without a hidden retry policy."""

    def execute(
        self,
        worker_path: str,
        config: JsonRecord,
        context: WorkerContext,
        resources: Mapping[str, float],
        *,
        on_poll: Callable[[], None] | None = None,
    ) -> JsonRecord:
        """Execute one attempt and report progress through the polling hook."""
        ...


class RetryableInfrastructureError(RuntimeError):
    """A backend confirmed worker/node loss. Application errors are not retried."""


def _worker_entry(worker_path: str, config: JsonRecord, context_dict: JsonRecord) -> JsonRecord:
    module, function = worker_path.split(":", 1)
    worker = getattr(importlib.import_module(module), function)
    result = worker(WorkerContext(**context_dict), config)
    if not isinstance(result, dict):
        raise ValueError("worker result must be a JSON mapping")
    return result


class LocalBackend:
    """Execute a worker in the driver's process."""

    def execute(
        self,
        worker_path: str,
        config: JsonRecord,
        context: WorkerContext,
        resources: Mapping[str, float],
        *,
        on_poll: Callable[[], None] | None = None,
    ) -> JsonRecord:
        """Run one attempt without retrying application errors."""
        return _worker_entry(worker_path, config, {**vars(context), "observer": on_poll})


class RayBackend:
    """Run one task attempt. The persistent supervisor owns retry decisions."""

    def __init__(self, *, address: str | None = None, poll_seconds: float = 0.5) -> None:
        import ray

        self.ray = ray
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be finite and positive")
        self.poll_seconds = poll_seconds
        if not ray.is_initialized():
            ray.init(address=address)

    def execute(
        self,
        worker_path: str,
        config: JsonRecord,
        context: WorkerContext,
        resources: Mapping[str, float],
        *,
        on_poll: Callable[[], None] | None = None,
    ) -> JsonRecord:
        """Run one attempt without retrying application errors."""
        remote = self.ray.remote(max_retries=0, retry_exceptions=False)(_worker_entry)
        try:
            reference = remote.options(**resources).remote(worker_path, config, vars(context))
            while True:
                ready, _ = self.ray.wait([reference], num_returns=1, timeout=self.poll_seconds)
                try:
                    if on_poll:
                        on_poll()
                except BaseException:
                    self.ray.cancel(reference, force=True)
                    raise
                if ready:
                    break
            result = self.ray.get(reference)
            if not isinstance(result, dict):
                raise ValueError("Ray worker result must be a JSON mapping")
            return result
        except (self.ray.exceptions.WorkerCrashedError, self.ray.exceptions.NodeDiedError) as exc:
            raise RetryableInfrastructureError(f"{type(exc).__name__}: {exc}") from exc
        except self.ray.exceptions.RayTaskError as exc:
            # Even an application-raised RetryableInfrastructureError is not
            # evidence of actual node/worker death.
            raise RuntimeError(f"Ray application task failed: {exc}") from exc
