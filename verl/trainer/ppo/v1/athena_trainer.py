# verl/trainer/ppo/v1/athena_trainer.py
import json
import logging
import os
import time
from typing import List

import pynvml
import torch

from verl.athena.controller import AthenaController, OptimizationAction
from verl.athena.runtime_state import (
    GPUStats,
    RolloutStats,
    RuntimeState,
    TrainingStats,
    TransferQueueStats,
    VLLMStats,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

logger = logging.getLogger(__name__)

_pynvml_initialized = False


@register_trainer("athena_trainer")
class AthenaTrainer(PPOTrainer):
    def __init__(self, config):
        super().__init__(config)
        self._athena_log_dir = config.trainer.get("athena_log_dir", "./athena_logs")
        os.makedirs(self._athena_log_dir, exist_ok=True)

        self._athena_log_file = open(
            os.path.join(self._athena_log_dir, f"trace_step_{os.getpid()}.jsonl"),
            "a",
        )

        self.controller = AthenaController(config=config, trainer=self)
        self._pending_actions: List[OptimizationAction] = []
        self._restart_request_path = os.path.join(self._athena_log_dir, "restart_requests.jsonl")

    def on_init_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)

    def get_llm_client(self):
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_train_begin(self):
        if self.config.skip.rollout_tq.enable:
            return
        num_warmup_batches = self.config.trainer.v1.colocate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def on_step_begin(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)
            self.checkpoint_manager.resume_generation_replicas()

    def on_sample_begin(self):
        pass

    def on_sample_end(self):
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int):
        # 1. 采样 batch
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()

        # ===== Athena 注入点（关键） =====
        # 在 batch 刚构造完、准备进入 prepare_micro_batches 之前
        if self._pending_actions:
            for action in self._pending_actions:
                self.controller.apply_hot_action(action, batch)
            self._pending_actions = []

        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        batch = self._balance_batch(batch, metrics=metrics)

        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        state = self._build_runtime_state(batch, metrics, timing_raw, global_steps, epoch)
        actions = self.controller.decide(state)

        for action in actions:
            if action.apply_immediately:
                self._pending_actions.append(action)
            elif action.requires_restart:
                with open(self._restart_request_path, "a") as request_file:
                    request_file.write(json.dumps(action.to_dict()) + "\n")
                logger.warning("[Athena] Restart probe requested: %s", action.to_dict())

    def _build_runtime_state(self, batch, metrics, timing_raw, global_steps, epoch):
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

        vllm_stats = VLLMStats()
        tq_stats = TransferQueueStats()
        rollout_stats = RolloutStats()

        try:
            batch_size = len(batch.tags) if hasattr(batch, "tags") else 1
        except Exception:
            batch_size = 1

        vllm_stats.batch_size = int(batch_size)
        vllm_stats.prefill_tokens = int(metrics.get("prompt_length/mean", 0.0) * batch_size)
        vllm_stats.decode_tokens = int(metrics.get("response_length/mean", 0.0) * batch_size)
        vllm_stats.num_preempted = int(metrics.get("num_preempted", 0))
        vllm_stats.kv_usage = metrics.get("vllm/kv_usage", 0.0)
        vllm_stats.prefix_cache_hit = metrics.get("vllm/prefix_cache_hit", 0.0)
        vllm_stats.num_evictions = int(metrics.get("vllm/num_evictions", 0))
        vllm_stats.scheduler_queue = int(metrics.get("vllm/scheduler_queue", 0))
        vllm_stats.num_running = int(metrics.get("vllm/num_running", 0))
        vllm_stats.num_waiting = int(metrics.get("vllm/num_waiting", 0))

        tq_stats.running = int(metrics.get("tq/running", 0))
        tq_stats.finished = int(metrics.get("tq/finished", 0))
        tq_stats.pending = int(metrics.get("tq/pending", 0))
        tq_stats.failure = int(metrics.get("tq/failure", 0))
        tq_stats.queue_length = tq_stats.running + tq_stats.pending
        tq_stats.latency_ms = float(metrics.get("tq/latency_ms", 0.0))

        rollout_stats.generate_latency_ms = timing_raw.get("gen", 0.0) * 1000
        rollout_stats.request_latency_ms = float(metrics.get("rollout/request_latency_ms", 0.0))
        rollout_stats.ttft_ms = float(metrics.get("rollout/ttft_ms", 0.0))
        rollout_stats.tpot_ms = float(metrics.get("rollout/tpot_ms", 0.0))
        rollout_stats.num_preempted = int(metrics.get("num_preempted", 0))
        rollout_stats.num_aborted = int(metrics.get("response/aborted_ratio", 0.0) * batch_size)

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
                "critic_loss": metrics.get("critic/loss", 0.0),
                "actor_loss": metrics.get("actor/loss", 0.0),
                "kl": metrics.get("actor/kl", 0.0),
            },
        )

        self._athena_log_file.write(state.to_json() + "\n")
        self._athena_log_file.flush()

        print(
            f"[Athena] Step {global_steps}: "
            f"Throughput {state.throughput:.1f} tok/s, "
            f"Actor {state.update_actor_time:.3f}s, "
            f"Mem {state.gpu.memory_peak_gb:.1f}GB"
        )
        return state

    def collect_gpu_stats(self, metrics) -> GPUStats:
        global _pynvml_initialized
        gpu = GPUStats()
        if not torch.cuda.is_available():
            return gpu

        gpu.memory_peak_gb = metrics.get("actor/perf/max_memory_allocated_gb", 0.0)

        try:
            if not _pynvml_initialized:
                pynvml.nvmlInit()
                _pynvml_initialized = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu.memory_used_gb = mem.used / 1024**3
            gpu.memory_reserved_gb = torch.cuda.memory_reserved() / 1024**3
            gpu.memory_allocated_gb = mem.used / 1024**3
            gpu.memory_utilization = mem.used / mem.total
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu.utilization = util.gpu / 100.0
            try:
                gpu.utilization = torch.cuda.utilization() / 100.0
            except Exception:
                pass
            gpu.power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            gpu.temperature_celsius = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
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

    def __del__(self):
        if hasattr(self, "_athena_log_file"):
            self._athena_log_file.close()
