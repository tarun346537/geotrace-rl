"""Checkpoint recovery, metric ingestion and alert accounting."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from ..config import RecoveryConfig, RuntimeConfig
from ..resilience import retry_delay
from ..types import JsonRecord
from .backends import Backend, RetryableInfrastructureError
from .checkpoint import CheckpointStore
from .context import WorkerContext
from .io import _json, _read_jsonl
from .ledger import RunLedger
from .telemetry import TelemetrySink

LOGGER = logging.getLogger(__name__)


class RecoverySupervisor:
    """Own the run ledger and retry confirmed worker/node loss."""

    def __init__(
        self,
        directory: str | Path,
        backend: Backend,
        *,
        sink: TelemetrySink | None = None,
        thresholds: Mapping[str, float] | None = None,
        recovery: RecoveryConfig | None = None,
        ledger_path: str | Path | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ledger = RunLedger(ledger_path or self.directory / "runs.sqlite3")
        self.backend, self.sink = backend, sink
        self.recovery = recovery or RuntimeConfig.from_env().recovery
        self.sleeper = sleeper
        self.thresholds = thresholds or {"physics_failure_fraction": 0.2, "kl_mean": 0.2}
        if any(
            not isinstance(key, str)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for key, value in self.thresholds.items()
        ):
            raise ValueError("alert thresholds must map names to finite numbers")

    def _alert(
        self, run_id: str, attempt: int, key: str, title: str, text: str, level: str = "WARN"
    ) -> None:
        self.ledger.event(
            run_id, key, "alert_requested", attempt, {"title": title, "text": text, "level": level}
        )
        if self.sink and not self.ledger.has_event(run_id, f"delivered:{key}"):
            try:
                self.sink.alert(title, text, level)
            except Exception as exc:
                self.ledger.event(
                    run_id,
                    f"delivery-failed:{key}",
                    "sink_failure",
                    attempt,
                    {"operation": "alert", "error": f"{type(exc).__name__}: {exc}"},
                )
            else:
                self.ledger.event(run_id, f"delivered:{key}", "alert_delivered", attempt, {})

    def reconcile(self, run_id: str, *, summary: bool = True) -> JsonRecord:
        """Ingest new worker telemetry and checkpoint progress."""
        state = self.ledger.state(run_id)
        run_dir = self.directory / run_id
        for attempt in range(1, state["attempt"] + 1):
            rows, offset, partial = _read_jsonl(
                run_dir / f"attempt-{attempt}" / "metrics.jsonl",
                self.ledger.cursor(run_id, attempt, "metrics"),
            )
            if partial:
                self.ledger.event(
                    run_id,
                    f"attempt:{attempt}:partial-metric",
                    "partial_telemetry_line",
                    attempt,
                    {},
                )
            for row in rows:
                if row["run_id"] != run_id or row["attempt"] != attempt:
                    raise ValueError("telemetry identity does not match its attempt directory")
                fresh = self.ledger.metric(row)
                if fresh and self.sink:
                    try:
                        self.sink.log(row)
                    except Exception as exc:
                        self.ledger.event(
                            run_id,
                            f"log-failed:{attempt}:{row['step']}",
                            "sink_failure",
                            attempt,
                            {"operation": "log", "error": f"{type(exc).__name__}: {exc}"},
                        )
                for name, limit in self.thresholds.items():
                    if name in row["metrics"] and row["metrics"][name] > limit:
                        self._alert(
                            run_id,
                            attempt,
                            f"threshold:{attempt}:{row['step']}:{name}",
                            f"GeoTrace {name} threshold",
                            f"run={run_id} attempt={attempt} step={row['step']} {name}={row['metrics'][name]} > {limit}",
                        )
            self.ledger.cursor(run_id, attempt, "metrics", offset)
            failures, event_offset, _ = _read_jsonl(
                run_dir / f"attempt-{attempt}" / "events.jsonl",
                self.ledger.cursor(run_id, attempt, "events"),
            )
            for event in failures:
                self._alert(
                    run_id,
                    attempt,
                    f"worker-event:{attempt}:{event['step']}:{event['kind']}",
                    "GeoTrace nonfinite training metric",
                    _json(event),
                    "ERROR",
                )
            self.ledger.cursor(run_id, attempt, "events", event_offset)
        manifest = CheckpointStore(run_dir / "checkpoints").latest(verify_payload=False)
        if manifest:
            self.ledger.checkpoint(run_id, manifest)
        for pending in self.ledger.pending_alerts(run_id):
            detail = json.loads(pending["detail"])
            self._alert(
                run_id,
                pending["attempt"],
                pending["event_key"],
                detail["title"],
                detail["text"],
                detail["level"],
            )
        return self.ledger.summary(run_id) if summary else {}

    def run(
        self,
        run_id: str,
        worker_path: str,
        config: JsonRecord,
        *,
        max_recoveries: int | None = None,
        resources: Mapping[str, float] | None = None,
    ) -> JsonRecord:
        """Resume a run with bounded retries for infrastructure loss."""
        max_recoveries = self.recovery.max_recoveries if max_recoveries is None else max_recoveries
        if not isinstance(max_recoveries, int) or max_recoveries < 0:
            raise ValueError("max_recoveries must be a nonnegative integer")
        resources = {"num_cpus": 1, "num_gpus": 0} if resources is None else dict(resources)
        if not resources or any(
            name not in {"num_cpus", "num_gpus", "memory"} or not math.isfinite(value) or value < 0
            for name, value in resources.items()
        ):
            raise ValueError("provide nonnegative num_cpus/num_gpus/memory resource requests")
        spec = {"worker": worker_path, "config": config, "resources": resources}
        self.ledger.register(run_id, spec)
        state = self.ledger.state(run_id)
        if state["status"] == "completed":
            completed = json.loads(state["result"])
            if not isinstance(completed, dict):
                raise ValueError("completed result must be an object. inspect the run ledger")
            return completed
        if state["status"] in {"failed", "running"}:
            raise ValueError(f"run is {state['status']}. inspect ledger before restarting")
        if state["attempt"] >= self.recovery.circuit_failure_limit:
            raise RetryableInfrastructureError(
                "recovery circuit is open. inspect the run ledger before restarting"
            )
        while self.ledger.state(run_id)["attempt"] <= max_recoveries:
            attempt = self.ledger.begin(run_id)
            context = WorkerContext(run_id, attempt, str(self.directory / run_id))
            manifest = context.store.latest()
            if manifest:
                self.ledger.event(
                    run_id,
                    f"attempt:{attempt}:restored",
                    "checkpoint_restored",
                    attempt,
                    {"step": manifest["step"], "sha256": manifest["sha256"]},
                )

            def _poll() -> None:
                self.reconcile(run_id, summary=False)

            try:
                result = self.backend.execute(
                    worker_path, config, context, resources, on_poll=_poll
                )
                self.reconcile(run_id)
                self.ledger.complete(run_id, attempt, result)
                return result
            except RetryableInfrastructureError as exc:
                try:
                    self.reconcile(run_id)
                except Exception as reconciliation_error:
                    self.ledger.fail(
                        run_id,
                        attempt,
                        f"recovery telemetry/checkpoint corruption: {reconciliation_error}",
                    )
                    raise reconciliation_error from exc
                self.ledger.interrupted(run_id, attempt, str(exc))
                self._alert(
                    run_id,
                    attempt,
                    f"interruption:{attempt}",
                    "GeoTrace worker interrupted",
                    str(exc),
                )
                if attempt > max_recoveries:
                    raise
                if attempt >= self.recovery.circuit_failure_limit:
                    self.ledger.event(run_id, "recovery:circuit-open", "circuit_open", attempt, {})
                    raise RetryableInfrastructureError(
                        "repeated worker loss opened the recovery circuit. inspect node health"
                    ) from exc
                delay = retry_delay(
                    attempt,
                    self.recovery.initial_backoff_seconds,
                    self.recovery.max_backoff_seconds,
                )
                LOGGER.warning(
                    "worker_recovery_backoff",
                    extra={"run_id": run_id, "attempt": attempt, "delay_seconds": delay},
                )
                self.sleeper(delay)
            except BaseException as exc:
                # Preserve the original failure and any intact worker telemetry.
                try:
                    self.reconcile(run_id)
                except Exception as reconciliation_error:
                    self.ledger.event(
                        run_id,
                        f"attempt:{attempt}:reconcile-failure",
                        "telemetry_failure",
                        attempt,
                        {"error": f"{type(reconciliation_error).__name__}: {reconciliation_error}"},
                    )
                self.ledger.fail(run_id, attempt, f"{type(exc).__name__}: {exc}")
                raise
        raise RetryableInfrastructureError(
            "recovery budget exhausted. run remains interrupted for inspection"
        )
