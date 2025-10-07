#!/usr/bin/env bash
# GRPO launch settings for Linux/CUDA, verl==0.5.0 and eight GPUs.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import importlib.metadata as m; assert m.version("verl") == "0.5.0", "use verl==0.5.0. reward configuration is pinned to this version"'
MODEL_PATH="${GEOTRACE_MODEL:?Set GEOTRACE_MODEL to your base model ID or model directory}"
MODE="${GEOTRACE_ROLLOUT_MODE:-tool_agent}"
ROLLOUT_ARGS=()
REWARD_FUNCTION=compute_score
if [[ "$MODE" == "tool_agent" ]]; then
    REWARD_FUNCTION=compute_agent_score
    if [[ "${GEOTRACE_REWARD_MODE:-process}" == "process" ]]; then
        : "${GEOTRACE_TEXT_PRM:?Set GEOTRACE_TEXT_PRM to the trained transformer process reward model directory}"
    fi
    ROLLOUT_ARGS+=(
        "data.return_raw_chat=True"
        "actor_rollout_ref.rollout.mode=async"
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO_DIR/configs/agent_loop.yaml"
    )
elif [[ "$MODE" != "single_turn" ]]; then
    echo 'GEOTRACE_ROLLOUT_MODE must be tool_agent or single_turn' >&2
    exit 2
fi
MONITOR_ARGS=(
    --run-id "${GEOTRACE_RUN_ID:?Set GEOTRACE_RUN_ID to a stable experiment ID}"
    --directory "${GEOTRACE_MONITOR_DIR:?Set GEOTRACE_MONITOR_DIR to durable storage visible at the same path on every node}"
    --ledger-file "${GEOTRACE_LEDGER_FILE:?Set GEOTRACE_LEDGER_FILE to durable coordinator-local SQLite storage}"
    --address "${GEOTRACE_RAY_JOB_ADDRESS:-http://127.0.0.1:8265}"
)
if [[ -n "${GEOTRACE_MONITOR_WANDB_PROJECT:-}" ]]; then
    MONITOR_ARGS+=(--wandb-project "$GEOTRACE_MONITOR_WANDB_PROJECT")
fi
python -m monitoring.verl "${MONITOR_ARGS[@]}" -- \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${GEOTRACE_DATA_DIR:-outputs/verl_data}/train.parquet" \
    data.val_files="${GEOTRACE_DATA_DIR:-outputs/verl_data}/validation.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=4 \
    custom_reward_function.path="$REPO_DIR/geotrace/verl_reward.py" \
    custom_reward_function.name="$REWARD_FUNCTION" \
    trainer.logger='[console]' \
    trainer.project_name=geotrace \
    trainer.experiment_name=grpo-7b \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.default_local_dir="${GEOTRACE_CHECKPOINT_DIR:-outputs/grpo_checkpoints}" \
    trainer.resume_mode=auto \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    "${ROLLOUT_ARGS[@]}" \
    "$@"
