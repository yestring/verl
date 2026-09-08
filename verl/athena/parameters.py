"""Athena parameter lifecycle and configuration helpers.

The registry is deliberately conservative: a parameter is hot only when VeRL
reads it from every batch or from the trainer config at every loop iteration.
Everything that changes workers, data loaders, process groups, or model state is
restart-required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping


class ParameterLifecycle(str, Enum):
    HOT = "hot"
    RESTART_REQUIRED = "restart_required"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    lifecycle: ParameterLifecycle
    description: str
    backends: tuple[str, ...] = ("fsdp", "megatron")


PARAMETERS: dict[str, ParameterSpec] = {
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": ParameterSpec(
        path="actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
        lifecycle=ParameterLifecycle.HOT,
        description="Per-step actor token budget when dynamic batching is enabled.",
    ),
    "trainer.save_freq": ParameterSpec(
        path="trainer.save_freq",
        lifecycle=ParameterLifecycle.HOT,
        description="Checkpoint cadence, read by the trainer on every step.",
    ),
    "trainer.test_freq": ParameterSpec(
        path="trainer.test_freq",
        lifecycle=ParameterLifecycle.HOT,
        description="Validation cadence, read by the trainer on every step.",
    ),
    "actor_rollout_ref.actor.entropy_coeff": ParameterSpec(
        path="actor_rollout_ref.actor.entropy_coeff",
        lifecycle=ParameterLifecycle.HOT,
        description="Actor entropy coefficient, propagated to the live worker loss configuration.",
    ),
    "actor_rollout_ref.actor.kl_loss_coef": ParameterSpec(
        path="actor_rollout_ref.actor.kl_loss_coef",
        lifecycle=ParameterLifecycle.HOT,
        description="Actor KL-loss coefficient, propagated to the live worker loss configuration.",
    ),
    "actor_rollout_ref.actor.optim.lr": ParameterSpec(
        path="actor_rollout_ref.actor.optim.lr",
        lifecycle=ParameterLifecycle.HOT,
        description="Actor learning rate, propagated to live optimizer and scheduler state.",
    ),
    "data.train_batch_size": ParameterSpec(
        path="data.train_batch_size",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes the stateful data loader and optimizer batch geometry.",
    ),
    "data.dataloader_num_workers": ParameterSpec(
        path="data.dataloader_num_workers",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Requires rebuilding the DataLoader worker pool.",
    ),
    "data.max_prompt_length": ParameterSpec(
        path="data.max_prompt_length",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes tokenized input shapes.",
    ),
    "data.max_response_length": ParameterSpec(
        path="data.max_response_length",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes rollout and training sequence shapes.",
    ),
    "actor_rollout_ref.actor.ppo_mini_batch_size": ParameterSpec(
        path="actor_rollout_ref.actor.ppo_mini_batch_size",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes PPO update batch geometry.",
    ),
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": ParameterSpec(
        path="actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes actor micro-batch construction.",
    ),
    "actor_rollout_ref.actor.use_dynamic_bsz": ParameterSpec(
        path="actor_rollout_ref.actor.use_dynamic_bsz",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes actor batching mode.",
    ),
    "actor_rollout_ref.model.enable_gradient_checkpointing": ParameterSpec(
        path="actor_rollout_ref.model.enable_gradient_checkpointing",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes the model execution graph.",
    ),
    "actor_rollout_ref.actor.fsdp_config.param_offload": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.param_offload",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes FSDP parameter placement.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.optimizer_offload",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes FSDP optimizer placement.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.fsdp_config.sharding_strategy": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.sharding_strategy",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes the FSDP parameter, gradient, and optimizer sharding policy.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.rollout.n": ParameterSpec(
        path="actor_rollout_ref.rollout.n",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes the number of rollout samples per prompt.",
    ),
    "actor_rollout_ref.rollout.gpu_memory_utilization": ParameterSpec(
        path="actor_rollout_ref.rollout.gpu_memory_utilization",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Rollout engine memory is allocated during engine startup.",
    ),
    "actor_rollout_ref.rollout.max_num_seqs": ParameterSpec(
        path="actor_rollout_ref.rollout.max_num_seqs",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes rollout scheduler concurrency.",
    ),
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes Megatron process groups.",
        backends=("megatron",),
    ),
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.pipeline_model_parallel_size",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Changes Megatron process groups.",
        backends=("megatron",),
    ),
    # ---- entropy kernel decision space (all restart-required: kernels are
    # JIT-compiled and selected at engine init, decided by the offline loop) ----
    "actor_rollout_ref.model.use_fused_kernels": ParameterSpec(
        path="actor_rollout_ref.model.use_fused_kernels",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Master switch for the fused linear-cross-entropy / entropy kernel path.",
    ),
    "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Entropy backward kernel strategy; one of SUPPORTED_BACKWARD_METHODS in verl.utils.kernel.kernels.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.fsdp_config.entropy_kernel_forward_fused": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.entropy_kernel_forward_fused",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Use the fused single-pass forward entropy kernel.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.fsdp_config.entropy_kernel_vocab_per_split": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.entropy_kernel_vocab_per_split",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Vocab-per-split tile for the entropy kernels; null keeps per-strategy defaults.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.fsdp_config.entropy_kernel_n_group": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.entropy_kernel_n_group",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="N-group size for grouped / owner-compute entropy strategies; null keeps defaults.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.fsdp_config.entropy_kernel_block_size_m": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.entropy_kernel_block_size_m",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="BLOCK_SIZE_M for the entropy kernels; null keeps per-family defaults.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.fsdp_config.entropy_kernel_block_size_n": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.entropy_kernel_block_size_n",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="BLOCK_SIZE_N for the entropy kernels; null keeps per-family defaults.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.fsdp_config.entropy_kernel_block_size_k": ParameterSpec(
        path="actor_rollout_ref.actor.fsdp_config.entropy_kernel_block_size_k",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="BLOCK_SIZE_K for the entropy kernels; null keeps per-family defaults.",
        backends=("fsdp",),
    ),
    "actor_rollout_ref.actor.megatron.entropy_kernel_backward_method": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.entropy_kernel_backward_method",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Entropy backward kernel strategy; one of SUPPORTED_BACKWARD_METHODS in verl.utils.kernel.kernels.",
        backends=("megatron",),
    ),
    "actor_rollout_ref.actor.megatron.entropy_kernel_forward_fused": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.entropy_kernel_forward_fused",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Use the fused single-pass forward entropy kernel.",
        backends=("megatron",),
    ),
    "actor_rollout_ref.actor.megatron.entropy_kernel_vocab_per_split": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.entropy_kernel_vocab_per_split",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Vocab-per-split tile for the entropy kernels; null keeps per-strategy defaults.",
        backends=("megatron",),
    ),
    "actor_rollout_ref.actor.megatron.entropy_kernel_n_group": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.entropy_kernel_n_group",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="N-group size for grouped / owner-compute entropy strategies; null keeps defaults.",
        backends=("megatron",),
    ),
    "actor_rollout_ref.actor.megatron.entropy_kernel_block_size_m": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.entropy_kernel_block_size_m",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="BLOCK_SIZE_M for the entropy kernels; null keeps per-family defaults.",
        backends=("megatron",),
    ),
    "actor_rollout_ref.actor.megatron.entropy_kernel_block_size_n": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.entropy_kernel_block_size_n",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="BLOCK_SIZE_N for the entropy kernels; null keeps per-family defaults.",
        backends=("megatron",),
    ),
    "actor_rollout_ref.actor.megatron.entropy_kernel_block_size_k": ParameterSpec(
        path="actor_rollout_ref.actor.megatron.entropy_kernel_block_size_k",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="BLOCK_SIZE_K for the entropy kernels; null keeps per-family defaults.",
        backends=("megatron",),
    ),
}


def get_parameter_spec(path: str) -> ParameterSpec:
    try:
        return PARAMETERS[path]
    except KeyError as exc:
        raise KeyError(f"Athena parameter is not registered: {path}") from exc


def get_by_path(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for key in path.split("."):
        if isinstance(value, Mapping):
            if key not in value:
                return default
            value = value[key]
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
    return value


def set_by_path(config: Any, path: str, value: Any) -> None:
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        if isinstance(target, MutableMapping):
            target = target.setdefault(key, {})
        elif hasattr(target, key):
            target = getattr(target, key)
        else:
            raise KeyError(f"Cannot set {path}: missing component {key}")
    if isinstance(target, MutableMapping):
        target[keys[-1]] = value
    elif hasattr(target, keys[-1]):
        setattr(target, keys[-1], value)
    else:
        raise KeyError(f"Cannot set {path}: missing leaf {keys[-1]}")


def validate_overrides(overrides: Mapping[str, Any], backend: str | None = None) -> None:
    for path in overrides:
        spec = get_parameter_spec(path)
        if spec.lifecycle == ParameterLifecycle.UNSUPPORTED:
            raise ValueError(f"Unsupported Athena parameter: {path}")
        if backend is not None and backend not in spec.backends:
            raise ValueError(f"{path} is not supported by backend {backend}; supported={spec.backends}")
