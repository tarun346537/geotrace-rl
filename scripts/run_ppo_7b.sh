#!/usr/bin/env bash
# Matched model/data baseline with PPO critic allocation.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${GEOTRACE_MODEL:?Set GEOTRACE_MODEL}"
bash "$REPO_DIR/scripts/run_grpo_7b.sh" \
    algorithm.adv_estimator=gae \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.optim.lr=1e-5 \
    trainer.experiment_name=ppo-7b \
    trainer.default_local_dir="${GEOTRACE_PPO_CHECKPOINT_DIR:-outputs/ppo_checkpoints}" \
    "$@"
