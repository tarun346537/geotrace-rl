"""Ray Jobs loss handling and native callback contracts."""

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_native_checkpoints import write_native

from geotrace.config import RecoveryConfig
from geotrace.ops.context import WorkerContext
from geotrace.ops.recovery import RecoverySupervisor
from geotrace.ops.verl_checkpoint import publish_native, restore_native
from geotrace.ops.verl_job import VerlJobBackend
from geotrace.types import JsonRecord
from geotrace.verl_driver import infrastructure_failure, run_attempt
from geotrace.verl_hooks import install_hooks, reward_metrics


class Jobs:
    """Synchronous fault injection behind the public Jobs client contract."""

    def __init__(self, root: Path, failure: str = "loss") -> None:
        self.root, self.failure = root, failure
        self.submissions: list[dict[str, Any]] = []
        self.polls = 0
        self.terminal = "FAILED"
        self.restored: list[int | None] = []

    def submit_job(
        self,
        *,
        submission_id: str,
        entrypoint: str,
        entrypoint_num_cpus: int,
        entrypoint_num_gpus: int,
        metadata: dict[str, str],
        runtime_env: JsonRecord,
    ) -> str:
        kwargs: JsonRecord = dict(
            submission_id=submission_id,
            entrypoint=entrypoint,
            entrypoint_num_cpus=entrypoint_num_cpus,
            entrypoint_num_gpus=entrypoint_num_gpus,
            metadata=metadata,
            runtime_env=runtime_env,
        )
        self.submissions.append(kwargs)
        self.polls = 0
        spec = json.loads(Path(shlex.split(kwargs["entrypoint"])[-1]).read_text())
        context = WorkerContext(**spec["context"])
        previous = restore_native(context)
        self.restored.append(previous["step"] if previous else None)

        def train() -> None:
            step = 1 if previous is None else previous["step"] + 1
            context.log_step(
                step, {"process_reward_mean": 0.7}, elapsed_seconds=10, allocated_gpus=2
            )
            write_native(self.root, step)
            publish_native(context, self.root, step, 2, False)
            if len(self.submissions) == 1:
                if self.failure == "loss":
                    context.log_step(
                        2, {"process_reward_mean": 0.2}, elapsed_seconds=10, allocated_gpus=2
                    )
                    raise ConnectionError("controlled node loss")
                raise ValueError("invalid tensor shape")

        try:
            run_attempt(context, train, lambda error: isinstance(error, ConnectionError))
            self.terminal = "SUCCEEDED"
        except (ConnectionError, ValueError):
            self.terminal = "FAILED"
        return str(kwargs["submission_id"])

    def get_job_status(self, job_id: str) -> str:
        self.polls += 1
        return "RUNNING" if self.polls == 1 else self.terminal

    def get_job_logs(self, job_id: str) -> str:
        return "controlled job output\n"

    def stop_job(self, job_id: str) -> bool:
        return True


def supervisor(root: Path, client: Jobs) -> RecoverySupervisor:
    return RecoverySupervisor(
        root,
        VerlJobBackend("test-address", client=client, sleeper=lambda delay: None),
        recovery=RecoveryConfig(initial_backoff_seconds=0, max_backoff_seconds=0),
    )


def test_native_job_restores_checkpoint_and_accounts_for_repeated_step(tmp_path: Path) -> None:
    client = Jobs(tmp_path / "native")
    monitor = supervisor(tmp_path / "runs", client)
    spec = {
        "overrides": ["algorithm.adv_estimator=grpo"],
        "environment": {},
        "working_directory": str(tmp_path),
        "python": "python",
    }
    result = monitor.run("experiment", "native", spec, resources={"num_gpus": 0})
    assert result["checkpoint"]["step"] == 2
    assert client.restored == [None, 1]
    assert all(row["entrypoint_num_gpus"] == 0 for row in client.submissions)
    assert [row["submission_id"] for row in client.submissions] == [
        "experiment-attempt-1",
        "experiment-attempt-2",
    ]
    report = monitor.reconcile("experiment")
    assert report["interruptions"] == 1
    assert report["resumed_attempts"] == 1
    assert report["observed_optimizer_steps"] == 3
    assert report["checkpoint_committed_fraction"] == pytest.approx(2 / 3)
    assert (tmp_path / "runs/experiment/attempt-1/job.log").is_file()
    assert (tmp_path / "runs/experiment/attempt-2/job-events.jsonl").is_file()


def test_native_application_error_does_not_trigger_another_job(tmp_path: Path) -> None:
    client = Jobs(tmp_path / "native", "application")
    monitor = supervisor(tmp_path / "runs", client)
    with pytest.raises(RuntimeError, match="invalid tensor shape"):
        monitor.run(
            "experiment",
            "native",
            {
                "overrides": [],
                "environment": {},
                "working_directory": str(tmp_path),
                "python": "python",
            },
            resources={"num_gpus": 0},
        )
    assert len(client.submissions) == 1
    assert monitor.ledger.state("experiment")["status"] == "failed"


def test_coordinator_poll_error_cancels_job_before_returning(tmp_path: Path) -> None:
    class Stoppable(Jobs):
        stopped = False

        def stop_job(self, job_id: str) -> bool:
            self.stopped, self.terminal = True, "STOPPED"
            return True

    client = Stoppable(tmp_path / "native")
    backend = VerlJobBackend("test", client=client, sleeper=lambda delay: None)

    def bad_poll() -> None:
        raise ValueError("corrupt telemetry")

    with pytest.raises(ValueError, match="corrupt telemetry"):
        backend.execute(
            "native",
            {
                "overrides": [],
                "environment": {},
                "working_directory": str(tmp_path),
                "python": "python",
            },
            WorkerContext("experiment", 1, str(tmp_path / "run")),
            {"num_gpus": 0},
            on_poll=bad_poll,
        )
    assert client.stopped
    assert client.get_job_status("experiment-attempt-1") == "STOPPED"


def test_unconfirmed_cancellation_requires_manual_reconciliation(tmp_path: Path) -> None:
    class Unresponsive(Jobs):
        def get_job_status(self, job_id: str) -> str:
            return "RUNNING"

    client = Unresponsive(tmp_path / "native")
    backend = VerlJobBackend("test", client=client, sleeper=lambda delay: None)

    def bad_poll() -> None:
        raise ValueError("poll failed")

    with pytest.raises(RuntimeError, match="reconcile job.json"):
        backend.execute(
            "native",
            {
                "overrides": [],
                "environment": {},
                "working_directory": str(tmp_path),
                "python": "python",
            },
            WorkerContext("experiment", 1, str(tmp_path / "run")),
            {"num_gpus": 0},
            on_poll=bad_poll,
        )
    assert len(client.submissions) == 1


def test_loss_classification_unwraps_ray_task_errors_but_not_value_errors() -> None:
    import ray.exceptions as errors

    loss = errors.NodeDiedError("node disappeared")
    wrapped = errors.RayTaskError("TaskRunner.run", "traceback", loss)
    assert infrastructure_failure(wrapped, errors)
    assert not infrastructure_failure(
        errors.RayTaskError("TaskRunner.run", "traceback", ValueError("shape")), errors
    )
    assert infrastructure_failure(errors.RayActorError(preempted=True), errors)
    assert not infrastructure_failure(errors.RayActorError(actor_init_failed=True), errors)


def test_native_hooks_log_real_step_duration_and_only_publish_completed_saves(
    tmp_path: Path,
) -> None:
    context = WorkerContext("experiment", 1, str(tmp_path / "run"))
    root = tmp_path / "native"

    class Tracking:
        def log(self, data: object, step: int, backend: object = None) -> None:
            return None

    class Trainer:
        global_steps = 2
        use_critic = False
        config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(root)))

        def _save_checkpoint(self) -> None:
            write_native(root, self.global_steps)

    module = SimpleNamespace(
        RayPPOTrainer=Trainer,
        compute_data_metrics=lambda batch, use_critic=True: {"actor/loss": 0.2},
    )
    install_hooks(Tracking, module, context, 2)
    install_hooks(Tracking, module, context, 2)
    batch = SimpleNamespace(
        non_tensor_batch={
            "physics_valid": [1, 0],
            "process_count": [2, 0],
            "process_sum": [1.4, 0],
            "process_min": [0.6, 0],
        }
    )
    metrics = module.compute_data_metrics(batch, use_critic=False)
    metrics.update({"training/global_step": 2, "timing_s/step": 4.25, "actor/kl_loss": 0.03})
    Tracking().log(metrics, step=2)
    row = json.loads((Path(context.directory) / "attempt-1/metrics.jsonl").read_text())
    assert row["elapsed_seconds"] == 4.25
    assert row["allocated_gpus"] == 2
    assert row["metrics"]["process_reward_mean"] == pytest.approx(0.7)
    assert row["metrics"]["physics_failure_fraction"] == 0.5
    assert row["metrics"]["kl_mean"] == 0.03
    Trainer()._save_checkpoint()
    assert restore_native(context)["step"] == 2
    assert reward_metrics(SimpleNamespace(non_tensor_batch={})) == {}
