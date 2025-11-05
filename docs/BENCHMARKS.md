# Benchmark protocols

[reference.json](../benchmarks/reference.json) records experiment values, units and measurement paths. Use it for metric definitions and use the benchmark command outputs for measurements of a new run.

## CPU, SimPEG and Ray workloads

Run from the repository root after the CPU setup in [METHODS.md](METHODS.md).

```bash
make check
python scripts/smoke_text_prm.py
python -m monitoring.check_ray
```

The current local check used Windows, Python 3.12, PyTorch 2.8.0 CPU, NumPy 2.2.6, SimPEG 0.24.0 and Ray 2.45.0. The default suite has one deliberately opt-in Ray case. The separate harness enables it and limits the subprocess to 30 seconds.

| Workload | Observed output | Evidence |
| --- | --- | --- |
| Unit/integration suite | 99 passed, 1 Ray case skipped | `tests/`. Rerun with `make coverage`. |
| Combined line/branch coverage | 68% | `outputs/review-coverage.json`, including unexecuted GPU/text-training entry points. |
| Real local Ray check | 9 passed | `outputs/ray-monitoring-smoke.log`. The worker exits after checkpoint 1 and replacement finishes 2. |
| Feature training resume | Bit-exact model and RNG state | `test_training.py`, `test_prm_training_resume.py`. |
| Transformer smoke | Two CPU optimizer steps, exported model reloaded | `outputs/text-prm-smoke/validation.json`. |
| SimPEG integration | Actual finite-cube forward, sign/units and shared labeling path passed | `test_optional_integrations.py`. |
| Parquet export | Survey-disjoint tables round-trip | `test_optional_integrations.py`. |

The transformer smoke checks trainer, export and inference interfaces with a tiny CPU model. Use the CUDA workflow in [METHODS.md](METHODS.md) for policy training and vLLM collection.

```bash
python -m geotrace.demo --seed 17 --samples 512 --output outputs/demo.json
```

This analytic workload uses 25 receivers and 100 candidate point-mass models. Its consistent-prefix MC scores are `[0.052734375, 0.1640625, 1.0]`, compared with exact probabilities `[0.05, 0.2, 1.0]`. The wrong-depth trace first becomes irrecoverable at zero-based step 1. A forged forward response receives −1.0. The objective arithmetic uses 9 active tokens and reports 0.0562296569.

## Process model and first-error metrics

The feature baseline commands in [METHODS.md](METHODS.md) produce `evaluation.json` and a separately loadable `best_model.pt`. The report includes soft-target MSE, error-positive step F1 and exact first-error trace accuracy. Training-only statistics normalize features. Validation selects the threshold applied unchanged to test surveys.

The textual PRM reports first-error location F1 only when independent per-step error annotations are present. A predicted first error at the wrong location counts as both a false positive and a false negative. A fully correct trace has no error location. The implementation is `text_prm_data.first_error_metrics`. `test_wrong_first_error_location_counts_as_fp_and_fn` fixes the denominator behavior.

Saved feature/text heads carry training/calibration survey identities and observation fingerprints. `geotrace.evaluate_text_prm` applies the exported validation threshold without refitting it and rejects training/calibration observations reused under another ID or in another file. Independent `error_label` annotations remain required for first-error scoring.

The 0.58 → 0.81 comparison uses first-error F1. Record the annotation source, validation/test survey IDs, threshold selection, model/checkpoint hash and number of traces for each evaluation. Report MC feasibility error and first-error location F1 under distinct metric names.

The exact 10-survey, 16-sample, two-epoch feature commands in `METHODS.md` ran four optimizer steps. Their test split has 8 traces / 24 prefixes. The observed soft-target MSE was 0.144059216, step-error F1 0.625, and exact first-error trace accuracy 0.25 at validation-selected threshold 0.455. The labeled input SHA-256 was `d85700d3897d1818a37cabcd1557bda7b48f268d1b82e8f9124f345365cd96f3`. These outputs cover feature-head training on the 10-survey fixture.

## Reward hacking and held-out prediction error

Add these fields to each evaluated textual rollout:

```json
{
  "evaluation_id": "heldout-case-001",
  "assigned_reward": 0.75,
  "heldout_survey": {
    "receivers_m": [-125, 125],
    "observations_microgal": [123.0, 123.0]
  }
}
```

This payload defines the input layout. Include the normal `survey_id`, `split: "test"`, `task` and `steps` fields. Keep evaluation receivers and observations out of policy inputs. Attach held-out observations and training-assigned rewards from the experiment records before evaluation.

```bash
python -m geotrace.benchmarks --baseline outcome-rollouts.jsonl --candidate process-rollouts.jsonl --high-reward-threshold 0.5 --output outputs/comparison.json
```

A hacked rollout has assigned reward at or above the threshold but fails independent final-outcome verification. Tool outputs are recomputed. The report includes the fraction over all rollouts and the fraction over high-reward rollouts. The 23% → 4% comparison uses the all-rollout denominator.

Prediction error is RMSE divided by held-out observation RMS. Evaluation receiver locations must be absent from policy inputs. Overlapping locations are rejected. Matched comparison requires identical evaluation IDs, tasks and held-out observations. An invalid or unsubmitted final model remains in the rollout denominator. Its prediction error is missing, and aggregate full-set error is null rather than computed on an easier subset. A valid-only diagnostic and coverage count are retained.

Use the rewards assigned by each training setup and record the verifier configuration for both runs. The outcome-only path includes tool-protocol verification, so matched comparisons must specify those checks alongside the reward model.

The 29% reduction is `1 - candidate_error / baseline_error` against outcome-only RL. Fix reward threshold and outcome tolerance before comparing policies.

## PPO versus GRPO peak memory

Measure each worker's actual CUDA peak allocated bytes during an agreed phase and aggregate the same devices. Record warmup, base model, precision, device type/count, context/response lengths, group size, batch sizes, training phase and software versions. Host-wide `nvidia-smi` memory usage is not the same metric as PyTorch allocated-byte peaks.

Each JSON input uses this shape:

```json
{
  "workload": {
    "model": "actual-model-id-and-revision",
    "hardware": "actual-device-type-and-count",
    "precision": "actual-precision",
    "batch_and_token_settings": "actual-matched-settings"
  },
  "peak_allocated_bytes_per_device": [123456789]
}
```

The scalar is a schema example. Replace it with recorded device peaks. PPO/GRPO method identity can be recorded outside the shared `workload` object. The comparator requires that object to be exactly equal and checks positive finite peaks for the same number of devices.

```bash
python -m geotrace.benchmarks --memory --baseline ppo-memory.json --candidate grpo-memory.json --output outputs/memory-comparison.json
```

Memory reduction is `1 - sum(candidate_peaks) / sum(baseline_peaks)`. The comparison is 35% for a 7B policy on one node with 8 A100 GPUs. Collect per-device allocated-byte peaks from both training jobs before running the comparator.

## Interrupted jobs, retained work and GPU hours

```bash
python -m monitoring.run --config configs/monitoring.json --directory outputs/monitored-runs
python -m monitoring.status --directory outputs/monitored-runs --run-id scalar-checkpoint-example --verify-checkpoint
```

Use the run ID in the configuration. The default scalar example allocates zero GPUs, so its GPU-hour totals are zero. The local Ray integration injects one worker loss for recovery validation.

For experiment-level accounting, store the allocation ledger's full reserved GPU hours and supply that value to health reporting:

```bash
python -m monitoring.status --directory "$DURABLE_RUN_DIRECTORY" --run-id "$RUN_ID" --verify-checkpoint --reserved-gpu-hours "$MEASURED_RESERVED_GPU_HOURS"
```

The denominator is the sum of allocated GPU count × full reservation duration, including startup, idle periods and repeated work. The numerator is unique checkpoint-committed optimizer-step GPU hours. The helper rejects nonfinite, nonpositive or smaller-than-committed reservations. Supply reservation totals from allocation records rather than estimating them from sparse telemetry.

The experiment restored 11 interrupted jobs with zero lost runs and 96% usable GPU hours per experiment. For comparable runs, retain the launched-run manifest, durable completion/recovery records, synchronized step times and allocation intervals. Aggregate recovered/completed identities across that manifest. [MONITORING.md](MONITORING.md) defines each ledger field.
