import importlib.util
import json
import os
import tempfile
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Never

from geotrace.monitoring import (
    CheckpointStore,
    LocalBackend,
    RecoverySupervisor,
    RetryableInfrastructureError,
    RunLedger,
    WorkerContext,
    sample_nvidia_smi,
)
from geotrace.types import JsonRecord


class RecordingSink:
    def __init__(self) -> None:
        self.logs, self.alerts = [], []

    def log(self, record: JsonRecord) -> None:
        self.logs.append(record)

    def alert(self, title: str, text: str, level: str) -> None:
        self.alerts.append((title, text, level))


class InterruptedBackend:
    """Explicit fault injection: first attempt loses one uncommitted step."""

    def __init__(self) -> None:
        self.calls = 0
        self.restored = []

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
        if self.calls == 1:
            context.log_step(
                1,
                {"process_reward_mean": 0.7, "physics_failure_fraction": 0.3},
                elapsed_seconds=2,
                allocated_gpus=2,
            )
            context.save_checkpoint(1, b'{"step":1,"optimizer":"included in real workers"}')
            context.log_step(2, {"process_reward_mean": 0.6}, elapsed_seconds=3, allocated_gpus=2)
            if on_poll:
                on_poll()
            raise RetryableInfrastructureError("controlled worker-loss injection")
        self.restored.append(context.restore()[0])
        context.log_step(2, {"process_reward_mean": 0.8}, elapsed_seconds=4, allocated_gpus=2)
        context.save_checkpoint(2, b'{"step":2}')
        context.log_step(3, {"process_reward_mean": 0.9}, elapsed_seconds=5, allocated_gpus=2)
        context.save_checkpoint(3, b'{"step":3}')
        if on_poll:
            on_poll()
        return {"step": 3}


class MonitoringTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / "outputs" / "test-temp"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_checkpoint_integrity_monotonicity_and_duplicate_publish(self) -> None:
        store = CheckpointStore(self.directory)
        first = store.save(step=1, attempt=1, payload=b"full-state")
        self.assertEqual(first, store.save(step=1, attempt=1, payload=b"full-state"))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            store.save(step=1, attempt=2, payload=b"different")
        with self.assertRaisesRegex(ValueError, "regresses"):
            store.save(step=0, attempt=2, payload=b"old")
        (self.directory / first["filename"]).write_bytes(b"corrupt!!!")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            store.load()

    def test_recovery_is_idempotent_and_accounts_for_lost_work(self) -> None:
        backend, sink = InterruptedBackend(), RecordingSink()
        supervisor = RecoverySupervisor(self.directory, backend, sink=sink)
        result = supervisor.run("trial", "unused:worker", {}, max_recoveries=1)
        self.assertEqual(result, {"step": 3})
        self.assertEqual(backend.restored, [1])
        report = supervisor.reconcile("trial")
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(report["interruptions"], 1)
        self.assertEqual(report["resumed_attempts"], 1)
        self.assertAlmostEqual(report["observed_step_gpu_hours"], 28 / 3600)
        self.assertAlmostEqual(report["checkpoint_committed_gpu_hours"], 22 / 3600)
        self.assertAlmostEqual(report["checkpoint_committed_fraction"], 22 / 28)
        self.assertEqual(
            len(sink.alerts), 2
        )  # One real threshold event and one injected interruption.
        self.assertEqual(len(sink.logs), 4)
        supervisor.reconcile("trial")
        self.assertEqual(len(sink.alerts), 2)
        self.assertEqual(supervisor.run("trial", "unused:worker", {}, max_recoveries=1), result)
        self.assertEqual(backend.calls, 2)
        # Persisted cursors and completion survive a fresh driver object.
        recreated = RecoverySupervisor(self.directory, backend, sink=sink)
        self.assertEqual(recreated.run("trial", "unused:worker", {}), result)
        self.assertEqual(recreated.reconcile("trial")["interruptions"], 1)
        self.assertEqual(len(sink.logs), 4)

    def test_application_error_is_not_retried(self) -> None:
        class Broken:
            calls = 0

            def execute(self, *args: object, **kwargs: object) -> JsonRecord:
                self.calls += 1
                raise ValueError("bad dataset")

        backend = Broken()
        supervisor = RecoverySupervisor(self.directory, backend)
        with self.assertRaisesRegex(ValueError, "bad dataset"):
            supervisor.run("bad-input", "unused:worker", {}, max_recoveries=3)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(supervisor.ledger.state("bad-input")["status"], "failed")

    def test_nan_metrics_raise_and_produce_explicit_error_alert(self) -> None:
        sink = RecordingSink()
        supervisor = RecoverySupervisor(self.directory, LocalBackend(), sink=sink)
        supervisor.ledger.register("nan", {})
        attempt = supervisor.ledger.begin("nan")
        context = WorkerContext("nan", attempt, str(self.directory / "nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            context.log_step(1, {"loss": float("nan")}, elapsed_seconds=1)
        supervisor.reconcile("nan")
        self.assertEqual(len(sink.alerts), 1)
        self.assertEqual(sink.alerts[0][2], "ERROR")

    def test_sink_failure_is_recorded_and_pending_alert_is_retried(self) -> None:
        class Flaky(RecordingSink):
            def __init__(self) -> None:
                super().__init__()
                self.requests = 0

            def alert(self, *args: object) -> None:
                self.requests += 1
                if self.requests == 1:
                    raise ConnectionError("offline")
                super().alert(*args)

        sink = Flaky()
        supervisor = RecoverySupervisor(self.directory, LocalBackend(), sink=sink)
        supervisor.ledger.register("alert", {})
        attempt = supervisor.ledger.begin("alert")
        context = WorkerContext("alert", attempt, str(self.directory / "alert"))
        context.log_step(1, {"kl_mean": 0.8}, elapsed_seconds=1)
        report = supervisor.reconcile("alert")
        self.assertEqual(report["alert_delivery_failures"], 1)
        self.assertEqual(len(sink.alerts), 1)
        supervisor.reconcile("alert")
        self.assertEqual(len(sink.alerts), 1)

    def test_truncated_last_line_is_accounted_for_without_accepting_corrupt_records(self) -> None:
        supervisor = RecoverySupervisor(self.directory, LocalBackend())
        supervisor.ledger.register("partial", {})
        attempt = supervisor.ledger.begin("partial")
        context = WorkerContext("partial", attempt, str(self.directory / "partial"))
        context.log_step(1, {"loss": 1.0}, elapsed_seconds=1)
        path = Path(context.directory) / "attempt-1" / "metrics.jsonl"
        with path.open("ab") as handle:
            handle.write(b'{"step":2')
        self.assertEqual(supervisor.reconcile("partial")["observed_optimizer_steps"], 1)
        self.assertTrue(supervisor.ledger.has_event("partial", "attempt:1:partial-metric"))
        with path.open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaises(json.JSONDecodeError):
            supervisor.reconcile("partial")

    def test_gpu_probe_uses_measured_values_and_handles_unavailable_driver(self) -> None:
        def runner(*args: object, **kwargs: object) -> SimpleNamespace:
            self.assertEqual(args[0][0], "nvidia-smi")
            return SimpleNamespace(stdout="0, GPU-abc, 72, 2048, 40960\n")

        measured = sample_nvidia_smi(runner)
        self.assertTrue(measured["available"])
        self.assertEqual(measured["devices"][0]["utilization_percent"], 72)

        def absent(*args: object, **kwargs: object) -> Never:
            raise FileNotFoundError("not installed")

        self.assertFalse(sample_nvidia_smi(absent)["available"])

    def test_running_state_requires_explicit_driver_reconciliation(self) -> None:
        ledger = RunLedger(self.directory / "runs.sqlite3")
        spec = {
            "worker": "unused:worker",
            "config": {},
            "resources": {"num_cpus": 1, "num_gpus": 0},
        }
        ledger.register("active", spec)
        attempt = ledger.begin("active")
        supervisor = RecoverySupervisor(self.directory, LocalBackend())
        with self.assertRaisesRegex(ValueError, "running"):
            supervisor.run("active", "unused:worker", {})
        self.assertTrue(
            ledger.interrupted("active", attempt, "operator confirmed old worker terminated")
        )
        self.assertFalse(ledger.interrupted("active", attempt, "duplicate notification"))
        self.assertEqual(ledger.summary("active")["interruptions"], 1)


@unittest.skipUnless(
    importlib.util.find_spec("ray") and os.environ.get("GEOTRACE_TEST_RAY") == "1",
    "Set GEOTRACE_TEST_RAY=1 to run Ray worker recovery",
)
class RayMonitoringTests(unittest.TestCase):
    def test_real_ray_recovers_worker_from_durable_checkpoint(self) -> None:
        import ray

        from geotrace.monitoring import RayBackend

        root = Path(__file__).resolve().parents[1] / "outputs" / "test-temp"
        root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=root) as directory:
                supervisor = RecoverySupervisor(directory, RayBackend())
                result = supervisor.run(
                    "ray-smoke",
                    "monitoring.example_worker:exit_worker_once_after_checkpoint",
                    {"steps": 2, "driver_pid": os.getpid()},
                    max_recoveries=1,
                )
                self.assertEqual(result["step"], 2)
                report = supervisor.reconcile("ray-smoke")
                self.assertTrue(report["completed"])
                self.assertEqual(report["interruptions"], 1)
                self.assertEqual(report["resumed_attempts"], 1)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()
