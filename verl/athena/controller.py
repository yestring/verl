"""Runtime controller for Athena's online control loop."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from verl.athena.parameters import ParameterLifecycle, get_by_path, get_parameter_spec, set_by_path
from verl.athena.runtime_state import RuntimeState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationAction:
    parameter: str
    value: Any
    reason: str
    lifecycle: ParameterLifecycle
    observed_step: int
    candidate_config: dict[str, Any] | None = None
    demo_steps: int = 0

    @property
    def name(self) -> str:
        """Compatibility alias used by older Athena integrations."""
        return self.parameter.rsplit(".", 1)[-1]

    @property
    def apply_immediately(self) -> bool:
        return self.lifecycle == ParameterLifecycle.HOT

    @property
    def requires_restart(self) -> bool:
        return self.lifecycle == ParameterLifecycle.RESTART_REQUIRED

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "reason": self.reason,
            "lifecycle": self.lifecycle.value,
            "observed_step": self.observed_step,
            "candidate_config": self.candidate_config,
            "demo_steps": self.demo_steps,
        }


class AthenaController:
    """Windowed, conservative controller.

    ``ppo_max_token_len_per_gpu`` is hot only with dynamic batching: VeRL
    consumes the value from each actor batch. Structural FSDP and batch-size
    changes are emitted as restart recommendations for the offline supervisor.
    """

    TOKEN_BUDGET_PATH = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
    SAVE_FREQ_PATH = "trainer.save_freq"
    TEST_FREQ_PATH = "trainer.test_freq"
    ENTROPY_PATH = "actor_rollout_ref.actor.entropy_coeff"
    KL_COEF_PATH = "actor_rollout_ref.actor.kl_loss_coef"
    LR_PATH = "actor_rollout_ref.actor.optim.lr"

    def __init__(self, config, trainer=None):
        self.config = config
        self.enabled = bool(get_by_path(config, "trainer.athena.enabled", True))
        self.window_size = int(get_by_path(config, "trainer.athena.window_size", 5))
        self.cooldown_steps = int(get_by_path(config, "trainer.athena.cooldown_steps", 5))
        self.high_memory = float(get_by_path(config, "trainer.athena.high_memory_threshold", 0.90))
        self.low_memory = float(get_by_path(config, "trainer.athena.low_memory_threshold", 0.70))
        self.low_utilization = float(get_by_path(config, "trainer.athena.low_gpu_utilization_threshold", 0.55))
        self.min_token_budget = int(get_by_path(config, "trainer.athena.min_token_budget", 1024))
        self.max_token_budget = int(get_by_path(config, "trainer.athena.max_token_budget", 32768))
        self.increase_factor = float(get_by_path(config, "trainer.athena.increase_factor", 1.25))
        self.decrease_factor = float(get_by_path(config, "trainer.athena.decrease_factor", 0.75))
        self.num_step_demo = int(get_by_path(config, "trainer.athena.num_step_demo", 20))
        self.current_token_budget = int(get_by_path(config, self.TOKEN_BUDGET_PATH, self.min_token_budget))
        self.dynamic_batching = bool(get_by_path(config, "actor_rollout_ref.actor.use_dynamic_bsz", False))
        self.trainer = trainer
        self.total_steps = int(get_by_path(config, "trainer.total_training_steps", 0) or 0)
        self.late_training_fraction = float(get_by_path(config, "trainer.athena.late_training_fraction", 0.70))
        self.target_kl = float(get_by_path(config, "trainer.athena.target_kl", 0.01))
        self.max_frequency = int(get_by_path(config, "trainer.athena.max_frequency", 100000))
        initial_lr = float(get_by_path(config, self.LR_PATH, 1e-6))
        self.min_lr = float(get_by_path(config, "trainer.athena.min_actor_lr", max(1e-8, initial_lr * 0.1)))
        self.max_lr = float(get_by_path(config, "trainer.athena.max_actor_lr", initial_lr * 2.0))
        self.grad_norm_lr_threshold = float(get_by_path(config, "trainer.athena.grad_norm_lr_threshold", 5.0))
        self._states: deque[RuntimeState] = deque(maxlen=max(1, self.window_size))
        self._last_hot_change_step = -self.cooldown_steps
        self._last_change_by_parameter: dict[str, int] = {}
        self._restart_recommendation_emitted = False

    def _window_average(self, attribute) -> float:
        return sum(attribute(state) for state in self._states) / len(self._states)

    def _hot_token_budget_action(self, state: RuntimeState) -> OptimizationAction | None:
        if not self.dynamic_batching or len(self._states) < self.window_size:
            return None
        if state.global_step - self._last_hot_change_step < self.cooldown_steps:
            return None

        avg_memory = self._window_average(lambda item: item.gpu.memory_utilization)
        avg_utilization = self._window_average(lambda item: item.gpu.utilization)
        new_budget = self.current_token_budget
        reason = ""
        if avg_memory >= self.high_memory:
            new_budget = max(self.min_token_budget, int(self.current_token_budget * self.decrease_factor))
            reason = f"mean GPU memory utilization {avg_memory:.2f} >= {self.high_memory:.2f}"
        elif avg_memory <= self.low_memory and avg_utilization <= self.low_utilization:
            new_budget = min(self.max_token_budget, int(self.current_token_budget * self.increase_factor))
            reason = f"mean GPU memory/utilization {avg_memory:.2f}/{avg_utilization:.2f} indicate available headroom"

        if new_budget == self.current_token_budget:
            return None
        self.current_token_budget = new_budget
        self._last_hot_change_step = state.global_step
        return OptimizationAction(
            parameter=self.TOKEN_BUDGET_PATH,
            value=new_budget,
            reason=reason,
            lifecycle=get_parameter_spec(self.TOKEN_BUDGET_PATH).lifecycle,
            observed_step=state.global_step,
        )

    def _ready(self, path: str, step: int) -> bool:
        return step - self._last_change_by_parameter.get(path, -self.cooldown_steps) >= self.cooldown_steps

    def _make_hot_action(self, path: str, value: Any, reason: str, step: int) -> OptimizationAction:
        self._last_change_by_parameter[path] = step
        return OptimizationAction(
            parameter=path,
            value=value,
            reason=reason,
            lifecycle=get_parameter_spec(path).lifecycle,
            observed_step=step,
        )

    def _decide_immediate_adjustments(self, state: RuntimeState) -> list[OptimizationAction]:
        """Return bounded changes that have a verified live application path."""
        if len(self._states) < self.window_size:
            return []
        actions: list[OptimizationAction] = []
        avg_memory = self._window_average(lambda item: item.gpu.memory_utilization)

        if avg_memory >= self.high_memory:
            for path in (self.SAVE_FREQ_PATH, self.TEST_FREQ_PATH):
                current = int(get_by_path(self.config, path, -1))
                # Negative disables the operation and is already the least I/O.
                if current > 0 and self._ready(path, state.global_step):
                    value = min(self.max_frequency, max(10, current * 2))
                    if value != current:
                        actions.append(
                            self._make_hot_action(
                                path,
                                value,
                                f"mean GPU memory utilization {avg_memory:.2f} >= {self.high_memory:.2f}",
                                state.global_step,
                            )
                        )

        progress = state.global_step / self.total_steps if self.total_steps > 0 else 0.0
        entropy = float(get_by_path(self.config, self.ENTROPY_PATH, 0.0))
        if (
            progress >= self.late_training_fraction
            and entropy > 0
            and self._ready(self.ENTROPY_PATH, state.global_step)
        ):
            actions.append(
                self._make_hot_action(
                    self.ENTROPY_PATH,
                    max(0.0, entropy * 0.5),
                    f"training progress {progress:.2%} reached late-stage annealing",
                    state.global_step,
                )
            )

        use_kl_loss = bool(get_by_path(self.config, "actor_rollout_ref.actor.use_kl_loss", False))
        observed_kl = float(state.training.ppo_kl)
        if use_kl_loss and observed_kl > 0 and self._ready(self.KL_COEF_PATH, state.global_step):
            current = float(get_by_path(self.config, self.KL_COEF_PATH, 0.001))
            value = current
            if observed_kl > self.target_kl * 1.5:
                value = min(10.0, current * 1.5)
            elif observed_kl < self.target_kl * 0.5:
                value = max(0.0, current * 0.8)
            if value != current:
                actions.append(
                    self._make_hot_action(
                        self.KL_COEF_PATH,
                        value,
                        f"observed PPO KL {observed_kl:.4g} versus target {self.target_kl:.4g}",
                        state.global_step,
                    )
                )

        grad_norm = float(state.training.grad_norm)
        if grad_norm > self.grad_norm_lr_threshold and self._ready(self.LR_PATH, state.global_step):
            current = float(get_by_path(self.config, self.LR_PATH, self.min_lr))
            value = max(self.min_lr, current * 0.8)
            if value != current:
                actions.append(
                    self._make_hot_action(
                        self.LR_PATH,
                        value,
                        f"gradient norm {grad_norm:.3g} exceeded {self.grad_norm_lr_threshold:.3g}",
                        state.global_step,
                    )
                )
        return actions

    def _entropy_kernel_engine_prefix(self) -> str:
        """Engine-config prefix for entropy kernel overrides."""
        strategy = str(get_by_path(self.config, "actor_rollout_ref.actor.strategy", "fsdp"))
        if strategy == "megatron":
            return "actor_rollout_ref.actor.megatron"
        return "actor_rollout_ref.actor.fsdp_config"

    def _cold_recommendation(self, state: RuntimeState) -> OptimizationAction | None:
        if self._restart_recommendation_emitted or len(self._states) < self.window_size:
            return None
        avg_memory = self._window_average(lambda item: item.gpu.memory_utilization)
        avg_utilization = self._window_average(lambda item: item.gpu.utilization)
        if avg_memory > self.low_memory or avg_utilization > self.low_utilization:
            return None

        use_fused_kernels = bool(get_by_path(self.config, "actor_rollout_ref.model.use_fused_kernels", False))
        if not use_fused_kernels:
            # Memory headroom is available: recommend the fused kernel path,
            # which trades memory for tensor-core throughput. The grouped
            # strategy bounds the d_logits buffer when headroom is limited.
            prefix = self._entropy_kernel_engine_prefix()
            candidate = {
                "actor_rollout_ref.model.use_fused_kernels": True,
                f"{prefix}.entropy_kernel_backward_method": "split_fused_n_grouped",
                f"{prefix}.entropy_kernel_forward_fused": True,
                f"{prefix}.entropy_kernel_vocab_per_split": 16384,
                f"{prefix}.entropy_kernel_n_group": 8,
            }
            parameter = "actor_rollout_ref.model.use_fused_kernels"
            self._restart_recommendation_emitted = True
            return OptimizationAction(
                parameter=parameter,
                value=True,
                reason=(
                    "sustained GPU and memory under-utilization; fused entropy kernels can trade memory "
                    "headroom for tensor-core throughput (verify with a restart probe)"
                ),
                lifecycle=get_parameter_spec(parameter).lifecycle,
                observed_step=state.global_step,
                candidate_config=candidate,
                demo_steps=self.num_step_demo,
            )

        path = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
        current = get_by_path(self.config, path, None)
        if current is None or self.dynamic_batching:
            return None
        candidate = {path: max(1, int(current) * 2)}
        self._restart_recommendation_emitted = True
        return OptimizationAction(
            parameter=path,
            value=candidate[path],
            reason="sustained GPU and memory under-utilization; verify a larger micro batch with a restart probe",
            lifecycle=get_parameter_spec(path).lifecycle,
            observed_step=state.global_step,
            candidate_config=candidate,
            demo_steps=self.num_step_demo,
        )

    def decide(self, state: RuntimeState) -> list[OptimizationAction]:
        if not self.enabled:
            return []
        self._states.append(state)
        actions = self._decide_immediate_adjustments(state)
        hot_action = self._hot_token_budget_action(state)
        if hot_action is not None:
            actions.append(hot_action)
        if actions:
            return actions
        cold_action = self._cold_recommendation(state)
        return [cold_action] if cold_action is not None else []

    def apply_hot_action(self, action: OptimizationAction, batch) -> None:
        if not action.apply_immediately:
            raise ValueError(f"Cannot apply non-hot action at runtime: {action.parameter}")
        if action.parameter == self.TOKEN_BUDGET_PATH:
            if not self.dynamic_batching:
                raise RuntimeError("ppo_max_token_len_per_gpu is hot only when actor.use_dynamic_bsz=true")
            batch.extra_info["max_token_len_per_gpu"] = int(action.value)
        elif action.parameter in (self.ENTROPY_PATH, self.KL_COEF_PATH, self.LR_PATH):
            if self.trainer is None or not hasattr(self.trainer, "actor_rollout_wg"):
                raise RuntimeError(f"{action.parameter} requires a live actor worker group")
            worker_key = {
                self.ENTROPY_PATH: "entropy_coeff",
                self.KL_COEF_PATH: "kl_loss_coef",
                self.LR_PATH: "lr",
            }[action.parameter]
            self.trainer.actor_rollout_wg.apply_actor_runtime_overrides({worker_key: float(action.value)})
        set_by_path(self.config, action.parameter, action.value)
        logger.info("[Athena] Applied hot parameter %s=%r", action.parameter, action.value)

    def apply_hot_overrides(self, overrides: Mapping[str, Any]) -> None:
        for path, value in overrides.items():
            spec = get_parameter_spec(path)
            if spec.lifecycle != ParameterLifecycle.HOT:
                raise ValueError(f"{path} requires restart")
            set_by_path(self.config, path, value)
