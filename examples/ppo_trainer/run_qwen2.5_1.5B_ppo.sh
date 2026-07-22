#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================
# 用户可调参数（适配双卡 A100）
# ============================
DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_PATH=${MODEL_PATH:-/home/hx/Qwen2.5-1.5B-Instruct}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}           # 双卡

# 批次大小（降低以适应显存）
train_batch_size=${TRAIN_BATCH_SIZE:-4}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-2} # 必须能被 train_batch_size 整除
max_prompt_length=${MAX_PROMPT_LENGTH:-128}
max_response_length=${MAX_RESPONSE_LENGTH:-128}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-1024}  # 初始小值，Athena 会动态调大

actor_lr=${ACTOR_LR:-1e-6}
critic_lr=${CRITIC_LR:-1e-5}
entropy_coeff=${ENTROPY_COEFF:-0}

# Megatron 并行度（搜索空间建议：TP=1/2, PP=1，因为只有双卡）
actor_tp=${ACTOR_TP:-1}
actor_pp=${ACTOR_PP:-1}
critic_tp=${CRITIC_TP:-1}
critic_pp=${CRITIC_PP:-1}

rollout_tp=${ROLLOUT_TP:-1}                  # vLLM 推理通常不需要 TP
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
rollout_n=${ROLLOUT_N:-1}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_ppo}
experiment_name=${EXPERIMENT_NAME:-qwen_1.5b_ppo_2gpu_athena}

gsm8k_train=${GSM8K_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
gsm8k_test=${GSM8K_TEST_FILE:-$HOME/data/gsm8k/test.parquet}

# ============================
# 固定参数数组（Megatron 后端）
# ============================
DATA=(
    algorithm.adv_estimator=gae
    data.train_files="['$gsm8k_train']"
    data.val_files="['$gsm8k_test']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    # Megatron 专用并行配置
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
)

CRITIC=(
    critic.model.path="$CRITIC_MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.optim.lr=${critic_lr}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    critic.megatron.tensor_model_parallel_size=${critic_tp}
    critic.megatron.pipeline_model_parallel_size=${critic_pp}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console"]'               # 可加 "wandb"
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    # ===== Athena 关键配置 =====
    trainer.use_v1=true
    trainer.v1.trainer_mode=athena_trainer
    +trainer.athena_log_dir=./athena_logs
)

EXTRA=(
    model_engine=megatron                          # 切换到 Megatron 后端
)

# ============================
# 启动训练
# ============================
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"