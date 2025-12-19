# Architecture

GeoTrace separates a gravity environment, a continuation policy, a process reward model, the policy optimizer, and experiment state. A successful forward prediction is an outcome. A learned prefix score is evidence about reaching that outcome under a particular continuation policy.

## Components and data flow

| Component | Source | Contract |
| --- | --- | --- |
| Survey ingestion | `data.py`, `data_cli.py` | Finite observations, explicit units, parameter grid, source URL/digest and survey-disjoint split. |
| Simulator | `physics.py`, `simpeg_adapter.py` | Recompute a complete model with the selected backend and reject invalid geometry or parameters. |
| Model transport | `model_client.py`, `resilience.py` | One complete tool action, bounded HTTP retries and endpoint circuit cooldown. |
| Trace collection | `rollouts.py`, `rollout_cli.py` | Execute real tools and retain reasoning, action, result and termination. |
| Process labels | `rollouts.label_trace` | Sample continuations from every prefix with pruning disabled and store successes and Wilson intervals. |
| Reward training | `text_prm*.py`, `train_prm.py`, `prm_*.py` | Soft-target training with grouped splits, early stopping, checkpoints and optional tracking. |
| Policy interaction | `agent_loop.py`, `verl_agent.py` | Preserve sampled token IDs. Observation positions have zero policy-loss mask. |
| Verified reward | `verl_reward.py` | Recompute tool protocol and bound process/outcome score. The actor owns KL. |
| Policy optimization | `scripts/run_grpo_7b.sh`, `run_ppo_7b.sh`, `verl_driver.py` | Pinned verl trainer, zero-GPU Ray Jobs driver, vLLM workers and committed native resume. |
| Evaluation | `benchmarks.py`, `prm_metrics.py`, `text_prm_data.py` | Fixed-threshold first-error metrics and matched held-out policy comparisons. |
| Operations | `ops/`, `monitoring/` | Immutable state, durable attempts, live telemetry, recovery, health and optional W&B. |

```mermaid
sequenceDiagram
    participant Policy
    participant Env as Gravity environment
    participant Labels as Continuation labeler
    participant PRM as Process reward head
    participant Actor as verl actor
    Policy->>Env: source model and reasoning
    Env-->>Policy: trusted forward output
    Labels->>Policy: resample from each stored prefix
    Policy->>Env: continuation tool calls
    Env-->>Labels: independent terminal outcome
    Labels->>PRM: prefix + soft probability target
    Actor->>Policy: sampled policy rollout
    Policy->>Env: tool calls
    Env-->>Actor: verified protocol
    PRM-->>Actor: bounded process/outcome score
```

## Gravity model and units

The lightweight backend is the analytic vertical response of a buried point mass. The SimPEG task backend is a finite uniform cube: one TensorMesh cell centered at the proposed x coordinate and depth, with all receivers on the normalized y=z=0 line. These are different forward models. Selecting `simpeg_cube` routes sampling, rewards, features and independent verification through the actual cube solve.

Mass divided by cube volume gives kg/m³. Dividing by 1000 gives SimPEG density contrast in g/cm³. SimPEG's upward-positive gz in mGal is negated and multiplied by 1000 for the downward-positive microGal task convention. Complete geometry and cube size key the cached unit-density response. The general multi-cell adapter accepts arbitrary receiver coordinates. Policy actions parameterize one source.

This compact single-source family keeps solver checks fast enough for repeated continuation labels. Survey geometry and priors constrain the non-unique inverse problem.

## Labels and learned process rewards

Textual traces may revise a source proposal after a forward call. The labeler executes the continuation policy repeatedly from each prefix and measures the successful-outcome fraction. Use Wilson intervals to interpret zero-success finite samples. Labels depend on policy, remaining turn budget, survey noise and task tolerance. Wilson intervals retain sampling uncertainty.

The finite-grid baseline instead commits one parameter assignment per step and samples unassigned values uniformly. Exact enumeration supplies an evaluation oracle on the small grid. Neither the exact feasibility flag nor MC targets are included in learned inputs.

The transformer renders observable survey/task information and reasoning/tool history, then predicts one scalar per prefix. It uses soft binary cross-entropy, optional backbone freezing, saved left-truncation/context settings and validation-based early stopping. Independent boolean error annotations permit first-error location F1. Selecting a threshold on test data is prohibited.

The eleven-feature MLP is a cheaper baseline. Its normalization comes from training data only. It evaluates step-error F1 and exact first-error trace accuracy against the finite-grid oracle. Use distinct metric names for the feature baseline and textual first-error location F1.

## GRPO, PPO and reward design

GRPO removes the learned value critic by centering and scaling rewards within each prompt group. The trade-off is several model samples per prompt and weak learning signal when a group has no reward variation. The default launcher samples four responses, uses rollout tensor parallelism two, and configures one node/eight GPUs. The PPO baseline changes to GAE and adds a critic. Memory comparisons must hold the rest of the workload fixed.

The reward parser checks model syntax, simulator output consistency and final submission before process scores are accepted. The process/outcome mixture is bounded. Incomplete and invalid trajectories have explicit penalties. Actor-side low-variance KL anchors the current policy to the reference model. The actor applies KL once.

The custom agent returns token-in/token-out trajectories to verl. It never re-tokenizes sampled policy tokens. New observation text is encoded once and masked from policy loss. Trailing observation-only tokens are removed so a terminal PPO penalty is attached to an actual response position. Step scores are aggregated into a scalar rollout reward for group-relative advantages.

The NumPy objective implementation checks clipping, group normalization, masked sequence weighting and the sampled KL estimator. Backpropagation and distributed optimization are performed by PyTorch/verl.

## State, recovery and failure modes

| Failure | Behavior |
| --- | --- |
| Malformed model action | Count a failed rollout/MC sample and retain an explicit termination reason. |
| Model service timeout, 429 or 5xx | Bounded exponential retries, then a visible transport error. Repeated failures open a cooldown circuit. |
| Model endpoint 4xx configuration error | Fail immediately rather than retry the same bad request. |
| Invalid solver arguments or forged result | Reject the tool history or issue a negative reward. |
| Dataset/split/context change on resume | Refuse a mismatched checkpoint definition. |
| Ray worker/node loss | Restore committed state and start a separately recorded attempt within the retry/circuit budget. |
| Application exception | Mark failure and propagate. Hidden Ray retries are disabled. |
| Partial trailing telemetry line | Exclude it and record the truncation. Malformed complete JSON fails visibly. |
| Checkpoint digest mismatch | Refuse restoration and mark the health report unhealthy. |
| W&B outage | Record sink failure and retry pending alerts on later reconciliation. |
| Driver disappearance | Require explicit old-worker reconciliation before another driver submits work. |

The feature trainer saves model, optimizer, RNGs, batch permutation/cursor and data identity. Its latest full resume state is distinct from the best validation weights exported for inference. Transformers Trainer supplies optimizer/scheduler/RNG checkpoints for the text path. Native verl checkpoints keep large sharded state out of the byte-oriented store: `ops.verl_checkpoint` commits a small reference after every rank and dataloader state have been saved. The native Ray Jobs backend uses the existing supervisor and reuses that reference as an explicit `resume_from_path`. `verl_hooks` records actual native optimizer durations and prefix-reward/physics/KL summaries from the trainer's logging boundary. [Native job contracts](MONITORING.md#native-verl-jobs) describe the pinned hooks and failure classification.

Checkpoint storage must outlive a worker. Ray tasks need the same durable path on every host. Only one supervisor owns a run. The SQLite ledger uses durable driver-local storage and one writer. The supervisor restores worker state after classified worker or node loss.

## Scaling and measurement boundaries

Ray can parallelize continuation jobs while vLLM batches inference on its host. The current collector and reward-training datasets remain in memory, and the reduced SimPEG environment uses a single-source parameter grid. Larger corpora need streaming and measured queue limits before increasing concurrency.

Measured step duration × reserved GPUs gives observed step GPU time. The ledger counts unique committed steps separately from repeated work. A full experiment fraction also requires external allocation start/end times and device counts: `gpu_time_accounting` divides committed GPU hours by that supplied reservation denominator and rejects impossible totals. GPU compute occupancy and useful-time fraction answer different questions.

[Benchmark protocols](BENCHMARKS.md), [data schemas](DATA.md) and [monitoring contracts](MONITORING.md) define the stored evidence.

Primary references: [GRPO paper](https://arxiv.org/abs/2402.03300), [objective reference](https://huggingface.co/docs/trl/grpo_trainer), [verl agent loop](https://verl.readthedocs.io/en/v0.5.x/advance/agent_loop.html), [SimPEG gravity API](https://docs.simpeg.xyz/latest/content/api/generated/simpeg.potential_fields.gravity.Simulation3DIntegral.html). TRL supplies the mathematical reference, while verl supplies the training runtime.
