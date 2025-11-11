"""Check recovery budgets, health inspection, and immutable checkpoint state."""

from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from geotrace.config import RecoveryConfig
from geotrace.monitoring import (
    CheckpointStore,
    LocalBackend,
    RecoverySupervisor,
    RetryableInfrastructureError,
    WorkerContext,
    gpu_time_accounting,
)
from geotrace.types import JsonRecord
from monitoring.status import run_health


class LosingBackend:
    """Inject a configured count of confirmed losses with real checkpoint writes."""

    def __init__(self, losses: int) -> None:
        self.losses = losses
        self.calls = 0
        self.restored_steps: list[int] = []

    def execute(
        self,
        worker_path: str,
        config: JsonRecord,
        context: WorkerContext,
        resources: Mapping[str, float],
        *,
        on_poll: Callable[[], None] | None = None,
    ) -> JsonRecord:
        self.calls += 1
        restored = context.restore()
        self.restored_steps.append(restored[0] if restored else 0)
        context.log_step(self.calls, {"loss": 1 / self.calls}, elapsed_seconds=1)
        context.save_checkpoint(self.calls, str(self.calls).encode())
        if self.calls <= self.losses:
            raise RetryableInfrastructureError("controlled test worker loss")
        return {"step": self.calls}


def test_recovery_backoff_and_durable_resume(run_directory: Path) -> None:
    backend = LosingBackend(losses=2)
    delays: list[float] = []
    config = RecoveryConfig(
        max_recoveries=2,
        initial_backoff_seconds=0.1,
        max_backoff_seconds=0.2,
        circuit_failure_limit=4,
    )
    supervisor = RecoverySupervisor(run_directory, backend, recovery=config, sleeper=delays.append)
    result = supervisor.run("retry", "unused:worker", {})
    assert result == {"step": 3}
    assert backend.restored_steps == [0, 1, 2]
    assert delays == [0.1, 0.2]
    state = run_health(run_directory, "retry", verify_checkpoint=True)
    assert state["healthy"]
    assert state["status"] == "completed"
    assert state["checkpoint_step"] == 3
    assert state["checkpoint_digest_verified"]
    assert state["measurements"]["interruptions"] == 2
    assert state["measurements"]["resumed_attempts"] == 2


def test_circuit_stops_repeated_worker_losses(run_directory: Path) -> None:
    backend = LosingBackend(losses=10)
    delays: list[float] = []
    config = RecoveryConfig(
        max_recoveries=9,
        initial_backoff_seconds=0.1,
        max_backoff_seconds=0.2,
        circuit_failure_limit=2,
    )
    supervisor = RecoverySupervisor(run_directory, backend, recovery=config, sleeper=delays.append)
    with pytest.raises(RetryableInfrastructureError, match="circuit"):
        supervisor.run("open", "unused:worker", {})
    assert backend.calls == 2
    assert delays == [0.1]
    assert supervisor.ledger.state("open")["status"] == "interrupted"
    assert supervisor.ledger.has_event("open", "recovery:circuit-open")
    with pytest.raises(RetryableInfrastructureError, match="circuit"):
        supervisor.run("open", "unused:worker", {})
    assert backend.calls == 2


def test_health_verifies_payload_and_missing_runs(
    run_directory: Path, fast_recovery: RecoveryConfig
) -> None:
    assert not run_health(run_directory, "absent")["healthy"]
    supervisor = RecoverySupervisor(run_directory, LocalBackend(), recovery=fast_recovery)
    result = supervisor.run("local", "monitoring.example_worker:optimize_scalar", {"steps": 3})
    assert result["step"] == 3
    healthy = run_health(run_directory, "local", verify_checkpoint=True)
    assert healthy["healthy"]
    assert healthy["measurements"]["interruptions"] == 0
    assert not run_health(run_directory, "unknown")["healthy"]
    store = CheckpointStore(run_directory / "local/checkpoints")
    manifest = store.latest()
    assert manifest is not None
    payload = store.root / manifest["filename"]
    original = payload.read_bytes()
    payload.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    corrupt = run_health(run_directory, "local", verify_checkpoint=True)
    assert not corrupt["healthy"]
    assert "mismatch" in corrupt["error"]


def test_metric_ranges_and_checkpoint_payload_are_validated(tmp_path: Path) -> None:
    context = WorkerContext("bounds", 1, str(tmp_path))
    for metrics in ({"physics_failure_fraction": 1.1}, {"gpu_utilization_percent": 101}):
        with pytest.raises(ValueError, match="must be in"):
            context.log_step(1, metrics, elapsed_seconds=1)
    with pytest.raises(ValueError, match="duration"):
        context.log_step(1, {"loss": 1.0}, elapsed_seconds=0)
    with pytest.raises(ValueError, match="nonempty"):
        context.save_checkpoint(1, b"")
    assert context.restore() is None


def test_reservation_accounting_uses_full_external_denominator(run_directory: Path) -> None:
    result = gpu_time_accounting(committed_gpu_hours=3, reserved_gpu_hours=4)
    assert result["usable_gpu_hours_fraction"] == 0.75
    assert result["reserved_gpu_hours"] == 4
    assert gpu_time_accounting(0, 4)["usable_gpu_hours_fraction"] == 0
    for committed, reserved in [(1, 0), (2, 1), (-1, 2), (1, float("inf")), (float("nan"), 2)]:
        with pytest.raises(ValueError, match="allocation timestamps"):
            gpu_time_accounting(committed, reserved)
    supervisor = RecoverySupervisor(run_directory, LosingBackend(0))
    supervisor.run("reservation", "unused:worker", {})
    report = run_health(run_directory, "reservation", reserved_gpu_hours=0.5)
    assert report["reservation_accounting"]["reserved_gpu_hours"] == 0.5
    assert report["reservation_accounting"]["usable_gpu_hours_fraction"] == 0
    invalid = run_health(run_directory, "reservation", reserved_gpu_hours=0)
    assert not invalid["healthy"]
    assert "positive reserved time" in invalid["error"]
