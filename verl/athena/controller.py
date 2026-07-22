"""Runtime controller for Athena's online control loop."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping

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
        self._states: Deque[RuntimeState] = deque(maxlen=max(1, self.window_size))
        self._last_hot_change_step = -self.cooldown_steps
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
            reason = (
                f"mean GPU memory/utilization {avg_memory:.2f}/{avg_utilization:.2f} indicate available headroom"
            )

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

    def _cold_recommendation(self, state: RuntimeState) -> OptimizationAction | None:
        if self._restart_recommendation_emitted or len(self._states) < self.window_size:
            return None
        avg_memory = self._window_average(lambda item: item.gpu.memory_utilization)
        avg_utilization = self._window_average(lambda item: item.gpu.utilization)
        if avg_memory > self.low_memory or avg_utilization > self.low_utilization:
            return None

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
        hot_action = self._hot_token_budget_action(state)
        if hot_action is not None:
            return [hot_action]
        cold_action = self._cold_recommendation(state)
        return [cold_action] if cold_action is not None else []

    def apply_hot_action(self, action: OptimizationAction, batch) -> None:
        if not action.apply_immediately:
            raise ValueError(f"Cannot apply non-hot action at runtime: {action.parameter}")
        if action.parameter == self.TOKEN_BUDGET_PATH:
            if not self.dynamic_batching:
                raise RuntimeError("ppo_max_token_len_per_gpu is hot only when actor.use_dynamic_bsz=true")
            batch.extra_info["max_token_len_per_gpu"] = int(action.value)
        set_by_path(self.config, action.parameter, action.value)
        logger.info("[Athena] Applied hot parameter %s=%r", action.parameter, action.value)

    def apply_hot_overrides(self, overrides: Mapping[str, Any]) -> None:
        for path, value in overrides.items():
            spec = get_parameter_spec(path)
            if spec.lifecycle != ParameterLifecycle.HOT:
                raise ValueError(f"{path} requires restart")
            set_by_path(self.config, path, value)
