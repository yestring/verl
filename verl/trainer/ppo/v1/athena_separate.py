# verl/trainer/ppo/v1/athena_separate.py
"""
Athena Trainer for Separate Async Mode

专门为 separate_async 模式设计的 Athena Trainer。
核心特性：
1. 继承自 PPOTrainerSeparateAsync，保留完整的异步调度能力
2. 在 on_step_end 中动态决定是否同步 standalone 副本权重
3. 保留完整的数据采集能力（RuntimeState + JSONL）
4. 控制器基于 off_policy_staleness 做出 skip 决策

使用方式：
    trainer.v1.trainer_mode: athena_separate
"""

import json
import os
import torch
import time
import logging
import pynvml
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync, register_trainer
from verl.athena.runtime_state import (
    RuntimeState,
    GPUStats,
    TrainingStats,
    VLLMStats,
    TransferQueueStats,
    RolloutStats,
)
from verl.athena.controller import AthenaController
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
_pynvml_initialized = False


@register_trainer("athena_separate")
class AthenaTrainerSeparate(PPOTrainerSeparateAsync):
    """
    Athena Trainer for Separate Async Mode

    在 PPOTrainerSeparateAsync 的基础上增加：
    1. Athena 数据采集（RuntimeState → JSONL）
    2. Athena 控制器（动态决策是否同步 standalone 权重）
    """

    def __init__(self, config):
        super().__init__(config)

        # ---------- Athena 日志 ----------
        self._athena_log_dir = config.trainer.get("athena_log_dir", "./athena_logs")
        os.makedirs(self._athena_log_dir, exist_ok=True)
        self._athena_log_file = open(
            os.path.join(self._athena_log_dir, f"trace_step_{os.getpid()}.jsonl"),
            "a"
        )

        # ---------- Athena 控制器 ----------
        self.controller = AthenaController(config.get("athena", {}).get("controller", {}))
        self._last_state = None

        # 统计 skip 决策
        self._sync_count = 0
        self._skip_count = 0

    # ============================================================
    # 生命周期钩子（继承自 PPOTrainerSeparateAsync，选择性重写）
    # ============================================================

    def on_step_end(self):
        """
        重写 on_step_end：加入控制器的 skip 逻辑。

        PPOTrainerSeparateAsync 的原始实现是：
            with marked_timer("update_weights", self.timing_raw):
                self.standalone_checkpoint_manager.update_weights(self.global_steps)

        现在由 Athena 控制器决定是否执行这个同步。
        """
        # 1. 控制器决策：是否同步 standalone 权重
        if self.controller.should_sync():
            with marked_timer("update_weights", self.timing_raw, color="red"):
                self.standalone_checkpoint_manager.update_weights(self.global_steps)
            self._sync_count += 1
            logger.debug(f"[Athena] Step {self.global_steps}: Standalone weight sync executed")
        else:
            self._skip_count += 1
            logger.info(f"[Athena] Step {self.global_steps}: Standalone weight sync SKIPPED (skip #{self._skip_count})")

        # 注意：不调用 super().on_step_end()，因为父类的 on_step_end 只有同步逻辑，
        # 我们已经用条件同步替代了它。

    # ============================================================
    # Athena 数据采集（与之前相同）
    # ============================================================

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        # 1. 父类计算 VeRL 指标
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        # 2. 采集 Athena 状态
        gpu_stats = self.collect_gpu_stats(metrics)

        training_stats = TrainingStats(
            entropy=metrics.get("actor/entropy", 0.0),
            grad_norm=metrics.get("actor/grad_norm", 0.0),
            lr=metrics.get("actor/lr", 0.0),
            ppo_kl=metrics.get("actor/ppo_kl", 0.0),
            clip_fraction=metrics.get("actor/pg_clipfrac", 0.0),
            pg_loss=metrics.get("actor/pg_loss", 0.0),
            kl_loss=metrics.get("actor/kl_loss", 0.0),
            actor_loss=metrics.get("actor/loss", 0.0),
            critic_loss=metrics.get("critic/loss", 0.0),
            reward=metrics.get("critic/rewards/mean", 0.0),
            advantage=metrics.get("critic/advantages/mean", 0.0),
            returns=metrics.get("critic/returns/mean", 0.0),
            mfu=metrics.get("perf/mfu/actor", 0.0),
        )

        vllm_stats, tq_stats, rollout_stats = self._collect_aux_stats(batch, metrics, timing_raw)

        state = RuntimeState(
            global_step=global_steps,
            epoch=epoch,
            timestamp=time.time(),
            gen_time=timing_raw.get("gen", 0.0),
            ref_time=timing_raw.get("ref", 0.0),
            values_time=timing_raw.get("values", 0.0),
            adv_time=timing_raw.get("adv", 0.0),
            update_actor_time=timing_raw.get("update_actor", 0.0),
            update_critic_time=timing_raw.get("update_critic", 0.0),
            update_weights_time=timing_raw.get("update_weights", 0.0),
            total_step_time=timing_raw.get("step", 0.0),
            throughput=metrics.get("perf/throughput", 0.0),
            total_num_tokens=int(metrics.get("perf/total_num_tokens", 0)),
            avg_prompt_len=metrics.get("prompt_length/mean", 0.0),
            avg_response_len=metrics.get("response_length/mean", 0.0),
            gpu=gpu_stats,
            training=training_stats,
            vllm=vllm_stats,
            transfer_queue=tq_stats,
            rollout=rollout_stats,
            extra={
                "off_policy_staleness": metrics.get(
                    "training/off_policy/trajectory_staleness/mean", 0.0
                ),
                "critic_loss": metrics.get("critic/loss", 0.0),
                "actor_loss": metrics.get("actor/loss", 0.0),
                "kl": metrics.get("actor/kl", 0.0),
            }
        )

        # 3. 写入 JSONL
        self._athena_log_file.write(state.to_json() + "\n")
        self._athena_log_file.flush()

        # 4. 控制台打印
        print(f"[Athena] Step {global_steps}: "
              f"Throughput {state.throughput:.1f} tok/s, "
              f"Actor {state.update_actor_time:.3f}s, "
              f"Gen {state.gen_time:.3f}s, "
              f"Mem {state.gpu.memory_peak_gb:.1f}GB")

        # 5. 控制器观察（基于当前 state）
        self._last_state = state.to_dict()
        self.controller.observe(self._last_state, global_steps)

    def _collect_aux_stats(self, batch, metrics, timing_raw):
        """收集 vLLM、TransferQueue、Rollout 辅助统计"""
        try:
            batch_size = len(batch.tags) if hasattr(batch, 'tags') else 1
        except:
            batch_size = 1

        vllm_stats = VLLMStats(
            batch_size=int(batch_size),
            prefill_tokens=int(metrics.get("prompt_length/mean", 0.0) * batch_size),
            decode_tokens=int(metrics.get("response_length/mean", 0.0) * batch_size),
            num_preempted=int(metrics.get("num_preempted", 0)),
            kv_usage=metrics.get("vllm/kv_usage", 0.0),
            prefix_cache_hit=metrics.get("vllm/prefix_cache_hit", 0.0),
            num_evictions=int(metrics.get("vllm/num_evictions", 0)),
            scheduler_queue=int(metrics.get("vllm/scheduler_queue", 0)),
            num_running=int(metrics.get("vllm/num_running", 0)),
            num_waiting=int(metrics.get("vllm/num_waiting", 0)),
        )

        tq_stats = TransferQueueStats(
            running=int(metrics.get("tq/running", 0)),
            finished=int(metrics.get("tq/finished", 0)),
            pending=int(metrics.get("tq/pending", 0)),
            failure=int(metrics.get("tq/failure", 0)),
            latency_ms=float(metrics.get("tq/latency_ms", 0.0)),
        )
        tq_stats.queue_length = tq_stats.running + tq_stats.pending

        rollout_stats = RolloutStats(
            generate_latency_ms=timing_raw.get("gen", 0.0) * 1000,
            request_latency_ms=float(metrics.get("rollout/request_latency_ms", 0.0)),
            ttft_ms=float(metrics.get("rollout/ttft_ms", 0.0)),
            tpot_ms=float(metrics.get("rollout/tpot_ms", 0.0)),
            num_preempted=int(metrics.get("num_preempted", 0)),
            num_aborted=int(metrics.get("response/aborted_ratio", 0.0) * batch_size),
        )

        return vllm_stats, tq_stats, rollout_stats

    # ============================================================
    # GPU 状态采集
    # ============================================================

    def collect_gpu_stats(self, metrics) -> GPUStats:
        global _pynvml_initialized

        gpu = GPUStats()

        if not torch.cuda.is_available():
            return gpu

        # VeRL metrics（整个 Step 峰值）
        gpu.memory_peak_gb = metrics.get("actor/perf/max_memory_allocated_gb", 0.0)

        # NVML 采集
        try:
            if not _pynvml_initialized:
                pynvml.nvmlInit()
                _pynvml_initialized = True

            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

            # Memory
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu.memory_used_gb = mem.used / 1024**3
            gpu.memory_reserved_gb = mem.used / 1024**3
            gpu.memory_allocated_gb = mem.used / 1024**3
            gpu.memory_utilization = mem.used / mem.total

            # GPU Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu.utilization = util.gpu / 100.0

            # Power
            gpu.power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            # Temperature
            gpu.temperature_celsius = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )

            # PCIe
            try:
                tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                gpu.pcie_tx_gb = tx / 1024 / 1024
                gpu.pcie_rx_gb = rx / 1024 / 1024
            except pynvml.NVMLError:
                pass

        except Exception:
            pass

        return gpu

    def get_controller_stats(self) -> dict:
        """返回控制器统计信息"""
        stats = self.controller.get_stats()
        stats["athena"] = {
            "sync_count": self._sync_count,
            "skip_count": self._skip_count,
            "skip_ratio": self._skip_count / (self._sync_count + self._skip_count + 1),
        }
        return stats

    def __del__(self):
        if hasattr(self, '_athena_log_file'):
            self._athena_log_file.close()