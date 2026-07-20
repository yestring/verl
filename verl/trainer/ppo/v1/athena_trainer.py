# verl/trainer/ppo/v1/athena_trainer.py
import json
import os
import torch
import time
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.athena.runtime_state import (
    RuntimeState,
    GPUStats,
    TrainingStats,
    VLLMStats,
    TransferQueueStats,
    RolloutStats,
)
from verl.utils.debug import marked_timer
import pynvml
import logging
from verl.athena.controller import AthenaController
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.trainer.amr_executor import AMRExecutor

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
            "a"
        )

        # ===== 修正1：先创建 AMRExecutor，再传给 Controller =====
        self.amr_executor = AMRExecutor(self, max_attempts=4)

        self.controller = AthenaController(
            config=config,
            trainer=self,
            amr_executor=self.amr_executor,
        )
        
        # 用于存储即时调整动作（max_token_len 等）
        self._pending_actions = {}

    # ---------- 生命周期钩子 ----------
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
        # 注意：这里不需要调用 apply_actions，因为我们在 _step 中注入 data

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)
            self.checkpoint_manager.resume_generation_replicas()

    def on_sample_begin(self):
        pass

    def on_sample_end(self):
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()


    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        # 复制自父类，添加注入逻辑
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

        # ===== Athena 注入 =====
        if "ppo_max_token_len_per_gpu" in self._pending_actions:
            new_len = self._pending_actions["ppo_max_token_len_per_gpu"]
            batch.extra_info["max_token_len"] = new_len
            if hasattr(self.config, 'actor_rollout_ref'):
                self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = new_len
            logger.info(f"[Athena] Injected max_token_len={new_len} into batch")
            del self._pending_actions["ppo_max_token_len_per_gpu"]

        # 2. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        # 3. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 4. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 5. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 7. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 8. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 9. update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch

    # ============================================================
    # 核心：重写训练 step 以注入即时调整
    # ============================================================
    def _step(self, data: TensorDict) -> dict:
        """
        重写父类的 _step 方法（假设父类有该方法）。
        如果没有，请找到实际调用 actor_engine.forward_backward_batch 的地方。
        """
        # 1. 注入即时调整（max_token_len）
        data = self._inject_athena_actions(data)
        
        # 2. 调用父类训练逻辑
        return super()._step(data)

    def _inject_athena_actions(self, data: TensorDict) -> TensorDict:
        """
        将 _pending_actions 中的 max_token_len 注入到 data 中，
        这样 VeRL 的 prepare_micro_batches 会自动识别并生效。
        """
        if "ppo_max_token_len_per_gpu" in self._pending_actions:
            new_len = self._pending_actions["ppo_max_token_len_per_gpu"]
            # 核心注入：写入 data，VeRL 会在切分微批次时读取
            tu.assign_non_tensor(data, "max_token_len", new_len)
            # 同时更新 config（保持同步，但不是必须）
            if hasattr(self.config, 'actor_rollout_ref'):
                self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = new_len
            logger.info(f"[Athena] Injected max_token_len={new_len} into current batch.")
            # 清空，避免重复注入
            del self._pending_actions["ppo_max_token_len_per_gpu"]
        return data

    # ============================================================
    # 指标采集与决策（保持不变）
    # ============================================================
    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        # 1. 先让父类计算标准指标
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        
        # 2. 构建 RuntimeState
        state = self._build_runtime_state(metrics, timing_raw, global_steps, epoch)
        
        # 3. 调用控制器决策（即时调整 + 探测内部触发）
        actions = self.controller.decide(state)
        
        # 4. 仅处理即时调整（暂存，供下个 step 注入）
        if "ppo_max_token_len_per_gpu" in actions:
            self._pending_actions["ppo_max_token_len_per_gpu"] = actions["ppo_max_token_len_per_gpu"]
            # 可选：打印日志
            logger.info(f"[Athena] Will adjust max_token_len to {actions['ppo_max_token_len_per_gpu']} next step")
        
        # 探测动作已被 Controller 内部消化，无需 Trainer 处理

    def _build_runtime_state(self, metrics, timing_raw, global_steps, epoch):
        # ---- GPU 状态 ----
        gpu_stats = self.collect_gpu_stats(metrics)

        # ---- 训练状态 ----
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
            batch_size = len(batch.tags) if hasattr(batch, 'tags') else 1
        except:
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
            }
        )

        self._athena_log_file.write(state.to_json() + "\n")
        self._athena_log_file.flush()

        print(f"[Athena] Step {global_steps}: "
              f"Throughput {state.throughput:.1f} tok/s, "
              f"Actor {state.update_actor_time:.3f}s, "
              f"Gen {state.gen_time:.3f}s, "
              f"Mem {state.gpu.memory_peak_gb:.1f}GB")
        
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
            gpu.memory_reserved_gb = mem.used / 1024**3
            gpu.memory_allocated_gb = mem.used / 1024**3
            gpu.memory_utilization = mem.used / mem.total
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu.utilization = util.gpu / 100.0
            try:
                gpu.utilization = torch.cuda.utilization() / 100.0
            except:
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
        if hasattr(self, '_athena_log_file'):
            self._athena_log_file.close()