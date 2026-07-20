import logging
import math
import threading
import weakref
from typing import Dict, Optional

from verl.athena.runtime_state import RuntimeState
from verl.trainer.amr_executor import AMRExecutor
logger = logging.getLogger(__name__)


class AthenaController:
    def __init__(self, config, trainer, amr_executor=None):
        self.config = config
        self._trainer_ref = weakref.ref(trainer)

        # ---------- 即时调整状态 ----------
        self.current_max_token_len = self._get_config_value(
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", default=24576
        )
        self._last_change_step = -10
        self._change_cooldown = 5

        # ---------- 探测状态 ----------
        self.probe_attempts = 0
        self.max_probe_attempts = 4
        self.consecutive_failures = 0
        self.probe_in_progress = False
        self.last_probe_trigger_step = -100

        # ---------- 调整历史 ----------
        self.adjustment_history = []

        # ---------- 执行器（由外部注入） ----------
        self.amr_executor = amr_executor

    def _get_trainer(self):
        trainer = self._trainer_ref()
        if trainer is None:
            raise RuntimeError("AthenaTrainer instance has been garbage collected")
        return trainer

    # ---- 辅助方法（与之前相同） ----
    def _get_config_value(self, path, default=None):
        keys = path.split(".")
        val = self.config
        for k in keys:
            if hasattr(val, k):
                val = getattr(val, k)
            elif isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def _set_config_value(self, path, value):
        keys = path.split(".")
        target = self.config
        for k in keys[:-1]:
            if hasattr(target, k):
                target = getattr(target, k)
            elif isinstance(target, dict):
                target = target.setdefault(k, {})
        last_key = keys[-1]
        if hasattr(target, last_key):
            setattr(target, last_key, value)
        elif isinstance(target, dict):
            target[last_key] = value

    # ============================================================
    # 即时调整
    # ============================================================
    def _decide_immediate_adjustments(self, state: RuntimeState) -> Dict:
        actions = {}
        current_step = state.global_step
        if current_step - self._last_change_step < self._change_cooldown:
            return actions

        if state.gpu.memory_utilization > 0.85:
            new_len = int(self.current_max_token_len * 0.7)
            new_len = max(new_len, 1024)
            actions["ppo_max_token_len_per_gpu"] = new_len
            actions["_reason"] = f"Memory high ({state.gpu.memory_utilization:.2f})"
            self._last_change_step = current_step
            return actions

        if state.training.mfu < 0.20 and state.gpu.memory_utilization < 0.70:
            new_len = int(self.current_max_token_len * 1.5)
            estimated_mem = state.gpu.memory_utilization * (new_len / self.current_max_token_len)
            if estimated_mem < 0.80:
                actions["ppo_max_token_len_per_gpu"] = new_len
                actions["_reason"] = f"MFU low ({state.training.mfu:.2f})"
                self._last_change_step = current_step
                self.adjustment_history.append({"step": current_step, "mfu": state.training.mfu})
        return actions

    # ============================================================
    # 探测决策 & 执行（直接调用 AMRExecutor）
    # ============================================================
    def _should_probe(self, state: RuntimeState) -> bool:
        if self.probe_in_progress:
            return False
        if self.probe_attempts >= self.max_probe_attempts:
            return False
        if self.consecutive_failures >= 2:
            return False
        if state.global_step - self._last_change_step < 10:
            return False
        if state.global_step - self.last_probe_trigger_step < 50:
            return False

        if state.training.mfu < 0.20 and state.gpu.memory_utilization < 0.75:
            recent = [h for h in self.adjustment_history if h["step"] > state.global_step - 10]
            if recent:
                avg_mfu = sum(h["mfu"] for h in recent) / len(recent)
                if avg_mfu < 0.22:
                    return True
            else:
                return True
        return False

    def _generate_candidate_config(self, state: RuntimeState) -> Dict:
        current_tp = self._get_config_value(
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size", default=2
        )
        current_pp = self._get_config_value(
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size", default=2
        )
        if current_tp < 8:
            strategy = "increase_tp"
        elif current_pp < 4:
            strategy = "increase_pp"
        else:
            strategy = "balance"
        return self._generate_candidate_config_for_tp_pp(current_tp, current_pp, strategy)

    def _generate_candidate_config_for_tp_pp(self, tp, pp, strategy):
        overrides = {}
        if strategy == "increase_tp" and tp < 8:
            new_tp = tp * 2
            overrides["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"] = new_tp
            overrides["actor_rollout_ref.ref.megatron.tensor_model_parallel_size"] = new_tp
            overrides["critic.megatron.tensor_model_parallel_size"] = new_tp
            overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"] = new_tp
        elif strategy == "increase_pp" and pp < 4:
            new_pp = pp * 2
            overrides["actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"] = new_pp
            overrides["actor_rollout_ref.ref.megatron.pipeline_model_parallel_size"] = new_pp
            overrides["critic.megatron.pipeline_model_parallel_size"] = new_pp
        elif strategy == "balance" and tp > 1 and pp > 1:
            overrides["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"] = pp
            overrides["actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"] = tp
            overrides["actor_rollout_ref.ref.megatron.tensor_model_parallel_size"] = pp
            overrides["actor_rollout_ref.ref.megatron.pipeline_model_parallel_size"] = tp
            overrides["critic.megatron.tensor_model_parallel_size"] = pp
            overrides["critic.megatron.pipeline_model_parallel_size"] = tp
            overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"] = pp
        return overrides

    def _calc_demo_steps(self, state: RuntimeState) -> int:
        total_steps = state.global_step or 100000
        step_time_s = state.total_step_time or 10.0
        step_factor = min(2.0, math.log10(max(total_steps, 1000)) / math.log10(1000))
        time_penalty = max(0.3, min(1.0, 10.0 / max(step_time_s, 0.1)))
        raw = int(12 * step_factor * time_penalty)
        return max(3, min(25, raw))

    # ============================================================
    # 主决策入口（现在包含探测执行）
    # ============================================================
    def decide(self, state: RuntimeState) -> Dict:
        """
        返回动作字典，包含：
          - "ppo_max_token_len_per_gpu": 即时调整
          - (探测动作现在直接在内部启动，不返回给 Trainer)
        """
        actions = {}

        # 1. 即时调整
        immediate = self._decide_immediate_adjustments(state)
        if immediate:
            actions.update(immediate)
            # 如果正在即时调整，暂不触发探测（避免双重变化）
            return actions

        # 2. 探测决策
        if self._should_probe(state):
            candidate = self._generate_candidate_config(state)
            demo_steps = self._calc_demo_steps(state)

            # ---- 直接调用 AMRExecutor（异步启动） ----
            if self.amr_executor is not None:
                logger.info(f"[Athena] Triggering probe: TP/PP change with {demo_steps} demo steps")
                self.probe_in_progress = True
                self.last_probe_trigger_step = state.global_step
                self.probe_attempts += 1

                baseline = {"throughput": state.throughput, "loss": state.training.actor_loss}

                # 异步执行探测（避免阻塞训练）
                def _run_probe():
                    try:
                        success = self.amr_executor.probe(
                            candidate_config=candidate,
                            demo_steps=demo_steps,
                            baseline_metrics=baseline,
                        )
                        # 回传结果
                        self._on_probe_result(success)
                    except Exception as e:
                        logger.error(f"[Athena] Probe execution failed: {e}")
                        self._on_probe_result(False)

                thread = threading.Thread(target=_run_probe)
                thread.daemon = True
                thread.start()
            else:
                logger.warning("[Athena] AMRExecutor not set; cannot perform probe.")

        return actions

    def _on_probe_result(self, success: bool):
        """探测完成后的回调"""
        self.probe_in_progress = False
        if success:
            self.consecutive_failures = 0
            logger.info("[Athena] Probe succeeded! Configuration committed.")
        else:
            self.consecutive_failures += 1
            logger.info("[Athena] Probe failed or not beneficial. Rolled back.")
        # 无论成败，探测尝试次数已增加，状态已更新

    def set_amr_executor(self, executor):
        """允许外部注入 AMRExecutor（便于测试）"""
        self.amr_executor = executor