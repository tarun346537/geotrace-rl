# Step monitoring and recovery contracts

`geotrace.monitoring` provides model-agnostic hooks for process-reward experiments.
It records real events and measurements. Counts and efficiency fractions are
computed from its durable ledger.

## Executable local example

From the repository root:

```bash
python -m monitoring.run --config configs/monitoring.json --directory outputs/monitored-runs
```

The example performs five scalar gradient steps, saving actual state after each
step. The scalar workload exercises the orchestration and checkpoint interfaces.
Calling it again with the same run ID/spec returns the saved completed result
without rerunning work. A changed specification requires a different run ID.

Inspect durable state without launching a worker:

```bash
python -m monitoring.status --directory outputs/monitored-runs --run-id scalar-checkpoint-example --verify-checkpoint
```

The health command checks SQLite integrity, run state and optional checkpoint
digest verification. Missing/failed/corrupt state returns a nonzero CLI status.
Interrupted state is retained with `needs_attention: true` for reconciliation.

For Ray, install a compatible Ray build in the training environment, make the
worker's Python module available on all workers, and add `--ray` (optionally
`--ray-address auto`). Example checkpoints and telemetry require a shared durable
filesystem mounted at the same absolute path on every participating worker. For
a single-host Ray check, an ordinary local directory is sufficient.

## Training hook

An importable `worker(context, config)` callable restores its full state, executes
training, reports each completed optimizer step, and returns a JSON-serializable
result. For PyTorch, include model, optimizer, scheduler, all relevant RNG states,
global step, dataloader/sampler state and minibatch cursor in the serialized
checkpoint. Model code supplies the checkpoint payload and restores its optimizer, RNG and sampler state.

```python
restored = context.restore()  # None or (committed_step, serialized_bytes)
# Restore your state and continue at the NEXT uncommitted global step.
context.log_step(
    step,
    {
        "process_reward_mean": float(mean_reward),
        "process_reward_min": float(min_reward),
        "physics_failure_fraction": float(rejected / rollouts),
        "kl_mean": float(kl),
        "loss": float(loss),
    },
    elapsed_seconds=measured_step_duration,
    allocated_gpus=actual_reserved_gpu_count,
)
context.save_checkpoint(step, serialized_full_state_bytes)
```

Step means the global optimizer step. Individual reasoning-step reward summaries
can be logged with names such as `process_reward_step_0_mean`. Keep the semantics
consistent across runs. Nonfinite or nonscalar values raise and produce a durable
error event. Duplicate telemetry for a given run/attempt/global step is accepted
only when its values agree. Any conflicting replay fails visibly.

Checkpoint payloads are immutable, named by step and SHA-256. An atomic latest
manifest is published after the payload is flushed. A resume verifies payload
size and digest. Checkpoints cannot regress or replace the state at an already
committed step with different bytes. The caller must ensure checkpoint storage
survives node loss. A worker's ephemeral local disk is insufficient for spot jobs.
The API serializes bytes in memory and targets a checkpoint-aware training worker.
Large sharded verl checkpoints continue to use verl's native checkpoint manager.

## Native verl jobs

The GRPO/PPO launchers call `monitoring.verl`, which submits `geotrace.verl_driver` through the Ray Jobs SDK. The driver reserves one CPU and **zero GPUs**. Native verl creates the GPU resource pool itself. Install the same pinned CUDA environment and repository at the same absolute path on the coordinator and all workers. Provision environments, models and datasets on the coordinator and workers before submitting the job.

For a stable CPU head plus one eight-GPU worker, start these in the corresponding environments:

```bash
# Stable head/coordinator host
ray start --head --num-gpus=0 --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265
# GPU worker host. HEAD_NODE is the reachable head address
ray start --address="$HEAD_NODE:6379" --num-gpus=8
```

Run the launch script from the coordinator with `GEOTRACE_RAY_JOB_ADDRESS=http://127.0.0.1:8265`. A same-host cluster also works for experiments, but losing that host loses its control plane. Head failover and replacement-node provisioning remain outside this supervisor.

`GEOTRACE_MONITOR_DIR` stores shared attempt payloads, checkpoint references and telemetry. `GEOTRACE_CHECKPOINT_DIR` (or the PPO-specific path) stores native sharded state. Both must survive GPU-node loss and be mounted at the same absolute paths everywhere. Put `GEOTRACE_LEDGER_FILE` on durable coordinator-local storage. A separate SQLite path avoids a network-filesystem writer. Do not relocate, mutate or delete input/model files within a running experiment.

The coordinator's run ID/specification is immutable. Each attempt has a deterministic submission ID, `job.json`, status transitions, driver logs, and a typed terminal record. Polling feeds measured events into the existing ledger and W&B sink. Submission/polling errors request job cancellation and wait up to 30 seconds for a terminal status. Unconfirmed cancellation stops with the submission ID for manual reconciliation. It never triggers an automatic duplicate job. On coordinator process loss, inspect the recorded Ray submission and confirm it has terminated before explicitly reconciling the ledger's running state. A Jobs API outage is not evidence that its job stopped.

Ray's public `worker_process_setup_hook` installs version-guarded verl 0.5 callbacks. `Tracking.log` supplies native `timing_s/step`, optimizer step, loss and actor KL. `compute_data_metrics` adds independently failed physics fraction and weighted reasoning-prefix reward summaries. Validation records go to a separate stream and do not count as optimizer work. Nonfinite step metrics stop the attempt. All reward-result fields are present on malformed as well as valid rollouts so native validation retains matched sample counts.

The checkpoint callback runs after native `_save_checkpoint` returns. It checks `data.pt`, role `fsdp_config.json`, and every rank's model, optimizer and extra-state file, including the critic for PPO. An atomic byte-store reference commits their path, topology and sizes. Restart uses `trainer.resume_mode=resume_path` and that explicit `global_step_N` path, ignoring a newer partial directory or tracker. Native PyTorch loading restores weights, optimizer, scheduler, RNG and dataloader state. The small reference is hashed. Large shards receive completeness/size checks, not independent content hashes.

Only unwrapped Ray `WorkerCrashedError`, `NodeDiedError`, `LocalRayletDiedError`, or `RayActorError.preempted` produce retryable terminal records. Shape/data/model errors, actor-constructor failures, cancellation, unknown driver death, and corrupt state stop for inspection. General actor death without confirmed preemption is deliberately not guessed from log text. Retry count, backoff and circuit limits use `GEOTRACE_CONFIG`.

Enable coordinator-owned W&B with `GEOTRACE_MONITOR_WANDB_PROJECT`. It resumes the same run ID and records attempts explicitly. Inspect a native run with its separate ledger:

```bash
python -m monitoring.status --directory "$GEOTRACE_MONITOR_DIR" --ledger-file "$GEOTRACE_LEDGER_FILE" --run-id "$GEOTRACE_RUN_ID" --verify-checkpoint
```

`test_native_checkpoints.py` and `test_verl_jobs.py` check actual connector code against small native-layout fixtures and controlled Jobs clients, including node-loss classification, incomplete saves, replay accounting and application failures. The fixtures exercise job submission and callbacks through controlled Jobs clients. The callback wraps pinned internal verl methods because that release has no checkpoint/metric callback registration interface. Changing verl requires checking these contracts again.

## Recovery policy

`RecoverySupervisor` owns a driver-local SQLite ledger. Ray retries are explicitly
disabled (`max_retries=0`, `retry_exceptions=False`) so attempts are accounted for
once. Only Ray `WorkerCrashedError` and `NodeDiedError` become retryable infrastructure
failures. A Python application exception, cancellation, invalid data, checkpoint
corruption or other Ray error is surfaced rather than endlessly retried. A task
can lose work after its last durable checkpoint. The next attempt restores the
checkpoint and starts from that progress.

`GEOTRACE_CONFIG=configs/recovery.toml` selects retry budgets, finite capped
exponential backoff and the repeated-loss circuit limit. Environment overrides
include `GEOTRACE_RECOVERY_RETRIES` and `GEOTRACE_RECOVERY_BACKOFF`. A circuit-open
run remains interrupted for inspection. Application failures never consume a
hidden infrastructure retry. Model HTTP retries use a separate cooldown circuit.

The supervisor polls append-only telemetry while the Ray task runs. File cursors
are persisted, so polls do not rescan an entire large trace history. A crash-truncated
last telemetry line is excluded and recorded explicitly. Malformed complete lines
cause an error. Driver-ledger corruption is not silently ignored.

Driver failure is a different failure domain. If a prior run is still marked
`running`, a new driver refuses to submit a duplicate worker. Confirm the old
worker is terminated, inspect its files, then explicitly mark that attempt
interrupted with `supervisor.ledger.interrupted(run_id, attempt, reason)` and
reconcile before resuming. This is not automatic driver/cluster-head failover.
Only one active supervisor may own a given run. Use a durable local database for
the driver. Do not share a SQLite writer database over NFS.

## W&B metrics and alerts

Pass an initialized W&B run through `WandbSink(run)`, or add `--wandb-project NAME`
to the example CLI. The default config alerts on `physics_failure_fraction > 0.2`
or `kl_mean > 0.2`. These are example thresholds, not tuned scientific constants.
Nonfinite metrics and worker interruptions also generate explicit alert events.
Thresholds are configurable in `configs/monitoring.json`.

The adapter uses `run.log` with explicit global-step/attempt fields and
`run.alert` with W&B `AlertLevel`. Recorded alert submissions are deduplicated in
the ledger. Failed sink calls are recorded. Pending alerts are retried on the
next poll. A driver crash between SDK submission and recording its acknowledgment
can cause an alert to be submitted twice. There is no cross-service transaction.
An interruption or alert request does not imply that a person received a message.
The ledger's `alert_delivered` event means the SDK call returned successfully.
Actual delivery depends on the W&B service and notification configuration.
Default runs have no W&B connection and send no external notifications.

Application logs use timestamp, level, logger, message, run ID and structured
extras such as step/attempt/metrics. `configs/logging.toml` records that field
contract. `GEOTRACE_LOG_LEVEL` and runtime TOML settings control the logger.

## GPU accounting

`sample_nvidia_smi()` queries measured utilization and GPU memory through
`nvidia-smi`. Missing drivers, unsupported readings and probe errors are explicit
`available: false` results. The query describes visible host devices. Select the
UUIDs actually allocated to the worker before averaging. Do not attribute other
jobs' devices to a run.

Ledger reports distinguish:

- `observed_step_gpu_hours`: sum of reported step duration × reserved GPUs,
  including work repeated after interruption.
- `checkpoint_committed_gpu_hours`: unique committed optimizer steps, counting
  each global step once even if attempts replay it.
- `checkpoint_committed_fraction`: committed/observed step GPU time, or null when
  no GPU time was reported.
- `observed_gpu_utilization_percent`: GPU-time-weighted mean of supplied measured
  utilization samples, or null when no such samples were supplied.

These step quantities exclude unreported work, scheduler wait and idle intervals.
They are not a reservation-wide useful-time fraction by themselves. Supply full
reserved GPU hours measured from allocation timestamps and device counts:

```bash
python -m monitoring.status --directory "$DURABLE_RUN_DIRECTORY" --run-id "$RUN_ID" --verify-checkpoint --reserved-gpu-hours "$MEASURED_RESERVED_GPU_HOURS"
```

`ops.accounting.gpu_time_accounting` divides unique checkpoint-committed GPU hours
by that external denominator and reports `usable_gpu_hours_fraction`. It rejects
nonfinite/nonpositive reservations and reservations smaller than committed work.
The denominator includes startup, idle and repeated work. It is never inferred
from sparse step logs. The default scalar example allocates zero GPUs, so its
GPU-hour totals are zero. GPU compute occupancy remains a separate measurement.

## Verification

`tests/test_monitoring.py` covers digest validation, checkpoint monotonicity,
controlled interruption/recovery, duplicate completion, durable cursor replay,
uncommitted work accounting, application-error handling, NaN alerts, failed alert
delivery and measured GPU probe parsing. Fault-injection tests use temporary
directories and injected worker failures.
`test_runtime.py` checks configuration, JSON context, HTTP retries and circuit
cooldown. `test_recovery_health.py` checks health, retry budgets and reservation
denominators. Experiment metrics are listed in `benchmarks/reference.json`.

Set `GEOTRACE_TEST_RAY=1` to opt into a real local-Ray worker integration test.
That test uses CPU resources, deliberately exits one worker after a durable
checkpoint, and verifies that a replacement worker completes from that state.
Use a GPU/cloud workload for allocation-wide performance measurements. The bounded
CPU harness `python -m monitoring.check_ray` limits startup/testing to 30 seconds
and retains its output in `outputs/ray-monitoring-smoke.log`.

Primary API references:

- [Ray task fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html).
- [Ray actor fault tolerance and idempotent replay](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html).
- [Ray WorkerCrashedError](https://docs.ray.io/en/latest/ray-core/api/doc/ray.exceptions.WorkerCrashedError.html).
- [W&B Run implementation and alert API](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py).
- [Ray 2.45 Jobs SDK and zero-GPU entrypoint resources](https://github.com/ray-project/ray/blob/ray-2.45.0/python/ray/dashboard/modules/job/sdk.py).
- [Ray 2.45 worker setup hook](https://github.com/ray-project/ray/blob/ray-2.45.0/python/ray/_private/runtime_env/setup_hook.py).
- [verl 0.5 native checkpoint and training log boundaries](https://github.com/verl-project/verl/blob/v0.5.0/verl/trainer/ppo/ray_trainer.py).
- [verl 0.5 FSDP checkpoint file contract](https://github.com/verl-project/verl/blob/v0.5.0/verl/utils/checkpoint/fsdp_checkpoint_manager.py).
