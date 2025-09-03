# Gravity and trace data contracts

Files use UTF-8 JSONL: one complete record per line. All traces from a survey belong to one split chosen before training. The readers reject duplicate trace IDs, changed task definitions under one survey ID, and survey IDs crossing train/validation/test. Identical receiver/observation pairs cannot cross splits under renamed IDs, including reordered CSV rows. These checks catch exact duplicates. Group neighboring surveys or shared geology together when those can leak correlated structure.

## Task and units

A task contains `receivers_m`, `observations_microgal`, `grid`, and optional `success_tolerance`, `forward_backend`, `cube_size_m`. The default solver is `point_mass`. `simpeg_cube` runs the finite uniform cube integral. Coordinates must be finite, observations must have positive RMS, grid values must be unique, and depths/masses must be positive.

Receivers lie on y=z=0. The source has y=0 and z=−depth. Gravity is downward-positive microGal. The cube must remain wholly below the receiver surface. Depth is its center depth, and positive mass divided by cube volume defines its density contrast. General multi-cell forward inputs are handled separately by `simpeg_adapter.py`.

The outcome is relative RMSE ≤ task tolerance, where relative RMSE divides prediction RMSE by observed RMS. A consistent simulator call can still fail that outcome.

## Assignment baseline

This is the input to `geotrace.data label`, feature training and survey-level parquet export. Each parameter is assigned once. Assignments are irrevocable.

```json
{
  "survey_id": "example-survey",
  "trace_id": "example-survey-0",
  "split": "train",
  "provenance": {"kind": "synthetic_example"},
  "task": {
    "receivers_m": [-100, 0, 100],
    "observations_microgal": [1708.724085999714, 2966.3555555555554, 1708.724085999714],
    "grid": {
      "x_m": [-50, 0, 50],
      "depth_m": [100, 150, 200],
      "mass_kg": [8000000000, 10000000000, 12000000000]
    },
    "success_tolerance": 0.035,
    "forward_backend": "point_mass",
    "cube_size_m": 50
  },
  "assignments": [["x_m", 0], ["depth_m", 150], ["mass_kg", 10000000000]]
}
```

Create an analytic point-mass sample with `python scripts/generate_sample_data.py --output outputs/sample-traces.jsonl`.

The labeling command adds `labeled_steps` with zero-based step, feature vector, MC successes/count/probability/Wilson bounds and an exact-feasibility flag. The exact flag is for finite-grid evaluation only. It is excluded from the learned features. Features contain assignment masks/normalized values, observable survey summaries and residuals from a deterministic default completion.

## Measured CSV import

Supply:

- A CSV with `x_m,gravity_microgal` columns in the stated coordinate/sign convention.
- A JSON object with exactly the `x_m`, `depth_m`, `mass_kg` grid arrays.
- A unique survey ID, split and source release URL.

```bash
python -m geotrace.data import-survey --survey observations.csv --grid parameter_grid.json --survey-id survey-A --split train --backend simpeg_cube --source-url "$SURVEY_RELEASE_URL" --output outputs/survey-A.jsonl
```

Choose a survey release and normalize its field coordinates before import. The importer records the CSV SHA-256 and source URL and constructs candidate assignments for labeling. Keep independent validation/test survey files and split membership fixed.

## Textual reasoning/tool traces

These records retain `survey_id`, `trace_id`, `split`, `task`, and provenance. Instead of `assignments`, they contain an ordered, nonempty `steps` list. Proposals can be revised. `submit_model` ends the trace.

Each step contains:

```json
{
  "reasoning": "Check the proposed source against the observations",
  "tool_call": {
    "name": "gravity_forward",
    "arguments": {
      "reasoning": "Check the proposed source against the observations",
      "model": {"x_m": 0, "depth_m": 150, "mass_kg": 10000000000}
    }
  },
  "tool_result": {
    "prediction_microgal": [1708.724085999714, 2966.3555555555554, 1708.724085999714],
    "relative_rmse": 0.0,
    "success": true,
    "forward_backend": "point_mass"
  },
  "call_id": "gravity-step-0"
}
```

The simulator creates results. The model proposes the tool name, reasoning and model. `rollouts.verify_steps` recomputes predictions, relative error and success, checks the backend and reasoning fields, and rejects actions after submission. The loader checks structure and label types. Labeling, training and reward paths perform the independent physics verification. Incorrect candidate models remain training examples when their simulator responses are faithful. Fabricated observations are rejected.

`rollouts label` appends `mc_label` per step: `successes`, `samples`, `probability`, `wilson_lower`, `wilson_upper`. A trace-level `labeling` record identifies model, seed, sample count and disabled pruning. The soft target is successful continuation probability under that sampled policy and turn budget.

Optional `error_label` is an independently supplied boolean per step. It supports first-error F1. Keep error annotations independent of the Monte Carlo sample count. The textual renderer includes only task information and reasoning/call/result fields, excluding targets and error annotations.

## verl contracts

`prepare_verl` writes one prompt per survey to train/validation/test parquet. The trainer uses validation for periodic checks. Test stays reserved. `extra_info` carries trusted task, survey identity and provenance. Tool-agent rows select `agent_name: geotrace_tool`.

The token loop expects one JSON action per turn, followed by a newline and `<END_STEP>`:

```json
{"reasoning": "Test this model", "tool": "gravity_forward", "model": {"x_m": 0, "depth_m": 150, "mass_kg": 10000000000}}
```

The environment appends `<TOOL_RESULT>...` observations with zero policy-loss mask. Sampled model tokens keep their IDs. `submit_model` is the terminal action. Malformed actions receive explicit failure handling.

The separate single-turn baseline expects only this complete response shape:

```json
{
  "assignments": [["x_m", 0], ["depth_m", 150], ["mass_kg", 10000000000]],
  "claimed_prediction_microgal": [1708.724085999714, 2966.3555555555554, 1708.724085999714]
}
```

Its claimed prediction must match the independent selected forward model. Unknown fields, duplicate/out-of-grid assignments, nonfinite arrays and inconsistent predictions receive a negative reward.

## Evaluation-only fields

Matched policy benchmarks add `evaluation_id`, `assigned_reward`, and `heldout_survey` at trace level. Held-out receiver locations must be disjoint from the model-visible task. The evaluator checks overlap with an absolute tolerance of 1e-8 m. [BENCHMARKS.md](BENCHMARKS.md) defines these records and their denominators.

Datasets, checkpoints and evaluation outputs are ignored by Git. Keep the source release, transformations, file digests, annotation protocol, split manifest and model revision alongside each actual experiment.
