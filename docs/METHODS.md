# Experiment workflow

Use separate CPU and Linux/CUDA environments. [Dependency profiles](../requirements/README.md) explain the exact pins and upstream compatibility constraints. The CPU workflow uses synthetic surveys. Supply measured surveys and model IDs to the CUDA workflow.

## CPU setup and finite-grid baseline

Python 3.12 is the locally checked full profile. Use `python -m venv .venv`, activate it, then:

```bash
python -m pip install -r requirements/cpu.txt -r requirements/dev.txt
python -m pip install -e . --no-deps
make check
python -m geotrace.data generate --surveys 10 --output outputs/surveys.jsonl
python -m geotrace.data label --input outputs/surveys.jsonl --samples 16 --output outputs/labels.jsonl
python -m geotrace.train_prm --input outputs/labels.jsonl --output outputs/prm --epochs 2 --synthetic
python -m geotrace.evaluate_prm --input outputs/labels.jsonl --checkpoint outputs/prm/best_model.pt --output outputs/prm/evaluation-only.json
```

The generator creates four assignment traces per synthetic survey and preassigns survey-disjoint train/validation/test splits. The feature model fits Monte Carlo soft targets. `best_model.pt` contains best validation inference weights and a validation-selected threshold. `checkpoint.pt` contains the last complete optimizer/RNG/cursor state.

For an intentional bounded training run followed by resume:

```bash
python -m geotrace.train_prm --input outputs/labels.jsonl --output outputs/resumable-prm --epochs 2 --max-steps 3 --synthetic
python -m geotrace.train_prm --input outputs/labels.jsonl --output outputs/resumable-prm --epochs 2 --resume outputs/resumable-prm/checkpoint.pt --synthetic
```

A matching data digest and training definition are required. Early stopping and optional `--wandb-project NAME` use the packaged [PRM defaults](../geotrace/prm_defaults.toml). Set `GEOTRACE_PRM_CONFIG` to a custom TOML path to override them. Results identify the actual input digest and workload.

## Import measured survey observations

The task environment expects a normalized gravity line with `x_m,gravity_microgal` CSV columns. Supply a JSON parameter grid, independent survey identities and the actual release URL. [DATA.md](DATA.md) defines units and coordinate limitations.

```bash
python -m geotrace.data import-survey --survey observations.csv --grid parameter_grid.json --survey-id survey-A --split train --backend simpeg_cube --source-url "$SURVEY_RELEASE_URL" --output outputs/survey-A.jsonl
```

Import independent validation and test surveys with their own split arguments. Combine records into `outputs/surveys.jsonl`. Do not move traces between splits afterward. Choose the source release before import. The importer records its digest and source URL, then creates candidate assignment traces for labeling.

The same task schema routes all prefix labels, features and verification through either `point_mass` or `simpeg_cube`. General density-mesh forward evaluation is available through `python -m geotrace.simpeg_adapter --help`.

## Linux/CUDA environment

Use Linux x86_64, Python 3.11, CUDA 12.4 with a matching NVIDIA driver/compiler, and sufficient GPUs for the selected launch configuration. Create a separate environment from the CPU profile.

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/cuda-verl.txt
python -m pip install -r requirements/cuda-kernels.txt --no-build-isolation
python -m pip install -e . --no-deps
python -m pip check
```

FlashAttention is a separate native build after PyTorch. Do not merge the newer CPU profile into this environment. See the [PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/) and [verl 0.5 installation contracts](https://verl.readthedocs.io/en/v0.5.x/start/install.html).

## Collect textual policy traces and continuation labels

Choose a base model supporting tool calls and its vLLM parser. In one terminal:

```bash
export GEOTRACE_MODEL="/path/to/tool-capable-model"
export TOOL_PARSER="parser-supported-by-your-model"
vllm serve "$GEOTRACE_MODEL" --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --port 8000
```

In another terminal in the same environment:

```bash
python -m geotrace.rollouts collect --input outputs/surveys.jsonl --model "$GEOTRACE_MODEL" --target-steps 140000 --output outputs/reasoning.jsonl
python -m geotrace.rollouts label --input outputs/reasoning.jsonl --model "$GEOTRACE_MODEL" --samples 16 --max-turns 8 --output outputs/labeled-reasoning.jsonl
```

Each step records reasoning, simulator arguments and the executed result. `gravity_forward` tests a proposed model. `submit_model` terminates a trajectory. The actual collected count appears in the adjacent manifest and may exceed the requested step count at the last trace boundary. Empty invalid attempts are counted in the collection manifest.

Every prefix is labeled by resampling model continuations with pruning disabled. Invalid sampled actions count as failed samples. Transport failures interrupt labeling rather than bias the target toward failure. Terminal submissions have deterministic outcomes. Add `--ray-workers 8 --ray-address auto` to labeling for bounded batches on an existing Ray cluster. `--ray-address local` starts a local instance.

## Train the textual process head

```bash
export PRM_BASE_MODEL="/path/to/sequence-classification-compatible-base"
python -m geotrace.text_prm --input outputs/labeled-reasoning.jsonl --model "$PRM_BASE_MODEL" --output outputs/text-prm --epochs 3 --batch-size 4 --max-length 2048 --checkpoint-steps 100
```

The single-output sequence-classification head fits soft binary cross-entropy. `--freeze-backbone` trains only the head. `--cpu` selects CPU execution. The renderer excludes MC targets and independent error annotations. Saved model metadata fixes input renderer, context length and left truncation for inference.

Transformers Trainer checkpoints include optimizer/scheduler/RNG state. Resume with `--resume outputs/text-prm/checkpoint-N`. The data digest and key training/context settings must match. Early stopping uses validation loss. Add `--wandb-project NAME` for optional tracking.

Independent boolean `error_label` values per step enable first-error location precision/recall/F1. Threshold selection uses validation traces only. The evaluation report includes soft-target error and computes first-error metrics when independent annotations are present.

Exported checkpoints retain training/calibration survey identities and observation fingerprints. Evaluate another file with the saved validation threshold:

```bash
python -m geotrace.evaluate_text_prm --input outputs/labeled-traces.jsonl --model outputs/text-prm/model --output outputs/text-prm/heldout.json
```

Independent error annotations are required to export that threshold. Evaluation applies the saved validation threshold and rejects training or calibration surveys reused as test data, including renamed observations in a different file.

`python scripts/smoke_text_prm.py` trains a tiny locally initialized transformer for two optimizer steps, exports it, reloads it, and verifies per-prefix inference without downloading weights.

## GRPO, PPO and outcome-only comparison

Prepare survey-level parquet and release the standalone collection server's GPU allocation before starting the trainer's vLLM workers.

```bash
python -m geotrace.prepare_verl --input outputs/surveys.jsonl --mode tool_agent --output outputs/verl_data
export GEOTRACE_MODEL="/path/to/base-7b-model"
export GEOTRACE_TEXT_PRM="$PWD/outputs/text-prm/model"
export GEOTRACE_MONITOR_DIR="/mnt/shared/geotrace/runs"
export GEOTRACE_LEDGER_FILE="$HOME/geotrace-state/runs.sqlite3"
export GEOTRACE_CHECKPOINT_DIR="/mnt/shared/geotrace/grpo-process-17/checkpoints"
export GEOTRACE_PPO_CHECKPOINT_DIR="/mnt/shared/geotrace/ppo-process-17/checkpoints"
export GEOTRACE_RUN_ID="grpo-process-17"
bash scripts/run_grpo_7b.sh
GEOTRACE_RUN_ID=ppo-process-17 bash scripts/run_ppo_7b.sh
GEOTRACE_RUN_ID=grpo-outcome-17 GEOTRACE_REWARD_MODE=outcome bash scripts/run_grpo_7b.sh trainer.experiment_name=grpo-outcome-only trainer.default_local_dir=/mnt/shared/geotrace/grpo-outcome-17/checkpoints
```

The one-node/eight-GPU configuration samples four responses per prompt and uses rollout tensor parallelism two. The custom agent preserves sampled token IDs and masks simulator observations. Process scores and terminal outcome form a scalar rollout reward. Actor KL is applied once by verl. A missing or broken text PRM in process mode stops training. Invalid policy output receives a negative reward instead.

Start the existing Ray cluster before running these commands. [Native job operations](MONITORING.md#native-verl-jobs) gives head/worker setup and storage assumptions. The launchers use Ray Jobs with a zero-GPU driver, while verl reserves its own GPU workers. The coordinator binds a stable run ID to the launch definition, retries confirmed infrastructure loss, and restores the last ledger-committed native checkpoint. Application errors stop the run.

`GEOTRACE_DATA_DIR`, `GEOTRACE_CHECKPOINT_DIR`, `GEOTRACE_PRM_DEVICE` and other runtime settings are listed in [.env.example](../.env.example). Set `GEOTRACE_MONITOR_WANDB_PROJECT` to send the coordinator's measured step metrics and alerts to one resumable W&B run. Keep the native trainer on its default console logger to avoid a second tracking run. The coordinator supplies native resume arguments from committed state.

PPO defaults to `outputs/ppo_checkpoints` even when the shell exports the GRPO checkpoint path. Override it with `GEOTRACE_PPO_CHECKPOINT_DIR` or `trainer.default_local_dir=...`. Keep separate directories for every algorithm, reward ablation, model and dataset. Automatic restoration assumes the same experiment definition.

For the assignment baseline use `prepare_verl --mode single_turn`, `GEOTRACE_ROLLOUT_MODE=single_turn`, and optional `GEOTRACE_PRM_CHECKPOINT=outputs/prm/best_model.pt`. This baseline can otherwise use seeded MC scoring. The multi-turn process path uses the text PRM.

## Evaluation and operations

[Benchmark protocols](BENCHMARKS.md) define matched held-out rollouts and memory records. [Monitoring](MONITORING.md) covers checkpoint storage, retry ownership, health checks, external reservation accounting, and W&B alerts.

[benchmarks/reference.json](../benchmarks/reference.json) records the experiment values. Retain survey, model and checkpoint identifiers together with allocation records for matched comparisons.
