"""Ray Jobs execution for native verl, with retries owned by the run ledger."""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast

from ..types import JsonRecord
from .backends import RetryableInfrastructureError
from .context import WorkerContext
from .io import _append, _atomic, _json, _utc
from .verl_checkpoint import restore_native


class JobClient(Protocol):
    """Public Ray Jobs SDK methods required by the coordinator."""

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
        """Submit a named job with explicit driver resource requests."""
        ...

    def get_job_status(self, job_id: str) -> object:
        """Read a submitted job's current lifecycle status."""
        ...

    def get_job_logs(self, job_id: str) -> str:
        """Fetch retained driver output after the job terminates."""
        ...

    def stop_job(self, job_id: str) -> bool:
        """Request cancellation of an outstanding submission."""
        ...


class VerlJobBackend:
    """Submit one zero-GPU driver job and retain durable attempt evidence."""

    def __init__(
        self,
        address: str,
        *,
        client: JobClient | None = None,
        poll_seconds: float = 1.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if client is None:
            from ray.job_submission import JobSubmissionClient

            client = cast(JobClient, JobSubmissionClient(address))
        if not 0 < poll_seconds <= 60:
            raise ValueError("job polling interval must be in (0,60] seconds")
        self.client, self.poll_seconds, self.sleeper = client, poll_seconds, sleeper

    def _cancel(self, job_id: str) -> None:
        self.client.stop_job(job_id)
        for _ in range(max(1, int(30 / self.poll_seconds))):
            if str(self.client.get_job_status(job_id)) in {"SUCCEEDED", "FAILED", "STOPPED"}:
                return
            self.sleeper(self.poll_seconds)
        raise RuntimeError(
            f"could not confirm Ray job {job_id} stopped. inspect it before restarting"
        )

    def execute(
        self,
        worker_path: str,
        config: JsonRecord,
        context: WorkerContext,
        resources: Mapping[str, float],
        *,
        on_poll: Callable[[], None] | None = None,
    ) -> JsonRecord:
        """Run a native job. Only its typed Ray loss record permits a retry."""
        if resources.get("num_gpus", 0) != 0:
            raise ValueError("the native job driver must reserve zero GPUs. verl owns GPU workers")
        restore_native(context)
        attempt = Path(context.directory) / f"attempt-{context.attempt}"
        attempt.mkdir(parents=True, exist_ok=True)
        launch = attempt / "launch.json"
        payload = {
            "context": {key: value for key, value in vars(context).items() if key != "observer"},
            "overrides": config["overrides"],
            "environment": config["environment"],
        }
        _atomic(launch, (_json(payload) + "\n").encode())
        job_id = f"{context.run_id}-attempt-{context.attempt}"
        entrypoint = f"cd {shlex.quote(config['working_directory'])} && " + shlex.join(
            [config["python"], "-m", "geotrace.verl_driver", str(launch)]
        )
        _atomic(
            attempt / "job.json",
            (_json({"submission_id": job_id, "entrypoint": entrypoint}) + "\n").encode(),
        )
        previous = None
        try:
            submitted = self.client.submit_job(
                submission_id=job_id,
                entrypoint=entrypoint,
                entrypoint_num_cpus=1,
                entrypoint_num_gpus=0,
                metadata={"run_id": context.run_id, "attempt": str(context.attempt)},
                runtime_env={"env_vars": config["environment"]},
            )
            if submitted != job_id:
                raise RuntimeError("Ray returned a different submission ID. inspect job.json")
            while True:
                status = str(self.client.get_job_status(job_id))
                if status != previous:
                    _append(attempt / "job-events.jsonl", {"status": status, "at_utc": _utc()})
                    previous = status
                if on_poll:
                    on_poll()
                if status not in {"PENDING", "RUNNING"}:
                    break
                self.sleeper(self.poll_seconds)
        except BaseException as error:
            try:
                self._cancel(job_id)
            except Exception as stop_error:
                raise RuntimeError(
                    f"can't confirm job {job_id} terminated. reconcile job.json before restarting: {stop_error}"
                ) from error
            raise
        _atomic(attempt / "job.log", self.client.get_job_logs(job_id).encode())
        name = "result.json" if status == "SUCCEEDED" else "failure.json"
        record_path = attempt / name
        if not record_path.exists():
            raise RuntimeError(
                f"native job {status.lower()} without {name}. inspect job.log before restarting"
            )
        record: JsonRecord = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("run_id") != context.run_id or record.get("attempt") != context.attempt:
            raise ValueError("native terminal record belongs to another attempt")
        if status == "SUCCEEDED":
            if restore_native(context) != record["checkpoint"]:
                raise ValueError("native completion differs from committed checkpoint state")
            return record
        if status == "FAILED" and record.get("retryable") is True:
            raise RetryableInfrastructureError(f"native Ray worker loss: {record['error']}")
        raise RuntimeError(f"native job {status.lower()}: {record.get('error', 'inspect job.log')}")
