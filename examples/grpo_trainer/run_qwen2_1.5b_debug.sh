#!/usr/bin/env bash
set -xeuo pipefail

########################################
# Debug Configuration
########################################

MODEL_PATH=${MODEL_PATH:-/home/hx/Qwen2.5-1.5B-Instruct}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="['$HOME/data/gsm8k/train.parquet']" \
    data.val_files="['$HOME/data/gsm8k/test.parquet']" \
    data.train_batch_size=16 \
    data.max_prompt_length=128 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024 \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.balance_batch=False \
    trainer.project_name=verl_debug \
    trainer.experiment_name=qwen2_1.5b_debug \
    trainer.logger='["console"]' \
    trainer.v1.trainer_mode=athena_trainer \
    +trainer.athena_log_dir=./athena_logs

