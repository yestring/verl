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
    "actor_rollout_ref.rollout.gpu_memory_utilization": ParameterSpec(
        path="actor_rollout_ref.rollout.gpu_memory_utilization",
        lifecycle=ParameterLifecycle.RESTART_REQUIRED,
        description="Rollout engine memory is allocated during engine startup.",
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
