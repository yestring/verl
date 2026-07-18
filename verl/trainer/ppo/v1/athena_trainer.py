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



    
    # ---------- 生命周期钩子 ----------
    def on_init_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)

    def get_llm_client(self):
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_train_begin(self):
        # 预热：提前放入 num_warmup_batches 个 batch
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
        # 根据控制器决策决定是否同步权重
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)
            self.checkpoint_manager.resume_generation_replicas()


        

    def on_sample_begin(self):
        pass

    def on_sample_end(self):
        # 采样结束后，中止所有未完成的请求并让副本休眠
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()

    # ---------- 核心：在指标计算完成后采集数据 ----------
    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        # 1. 先让父类计算所有 VeRL 指标（填充 metrics）
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)



        # 2. 现在 metrics 已包含完整数据，开始采集 Athena 状态
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
        
        # ---- 其他子结构先留空，后续扩展 ----
        vllm_stats = VLLMStats()
        tq_stats = TransferQueueStats()
        rollout_stats = RolloutStats()


        # ============================================================
        # vLLM Stats
        # ============================================================
        
        # 获取 batch size
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
        
        # ============================================================
        # Transfer Queue Stats
        # ============================================================
        
        tq_stats.running = int(metrics.get("tq/running", 0))
        tq_stats.finished = int(metrics.get("tq/finished", 0))
        tq_stats.pending = int(metrics.get("tq/pending", 0))
        tq_stats.failure = int(metrics.get("tq/failure", 0))
        tq_stats.queue_length = tq_stats.running + tq_stats.pending
        tq_stats.latency_ms = float(metrics.get("tq/latency_ms", 0.0))
        
        # ============================================================
        # Rollout Stats
        # ============================================================
        
        rollout_stats.generate_latency_ms = timing_raw.get("gen", 0.0) * 1000
        rollout_stats.request_latency_ms = float(metrics.get("rollout/request_latency_ms", 0.0))
        rollout_stats.ttft_ms = float(metrics.get("rollout/ttft_ms", 0.0))
        rollout_stats.tpot_ms = float(metrics.get("rollout/tpot_ms", 0.0))
        rollout_stats.num_preempted = int(metrics.get("num_preempted", 0))
        rollout_stats.num_aborted = int(metrics.get("response/aborted_ratio", 0.0) * batch_size)




        
        # ---- 组装 RuntimeState ----
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
                # 可继续添加 VeRL 的其他关键字段
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





    def collect_gpu_stats(self,metrics) -> GPUStats:
        global _pynvml_initialized
    
        gpu = GPUStats()
    
        if not torch.cuda.is_available():
            return gpu
    
        #########################################
        # VeRL metrics（整个Step峰值）
        #########################################
    
        gpu.memory_peak_gb = metrics.get(
            "actor/perf/max_memory_allocated_gb",
            0.0,
        )
    
        #########################################
        # NVML
        #########################################
    
        try:
    
            if not _pynvml_initialized:
                pynvml.nvmlInit()
                _pynvml_initialized = True
    
            handle = pynvml.nvmlDeviceGetHandleByIndex(
                torch.cuda.current_device()
            )
    
            ##########################
            # Memory
            ##########################
    
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
            gpu.memory_used_gb = mem.used / 1024**3
    
            gpu.memory_reserved_gb = mem.used / 1024**3
    
            gpu.memory_allocated_gb = mem.used / 1024**3
    
            gpu.memory_utilization = mem.used / mem.total
    
            ##########################
            # GPU Util
            ##########################
    
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    
            gpu.utilization = util.gpu / 100.0

            try:
                gpu.utilization = torch.cuda.utilization() / 100.0 #peak
            except:
                pass
    
            ##########################
            # Power
            ##########################
    
            gpu.power_watts = (
                pynvml.nvmlDeviceGetPowerUsage(handle)
                / 1000.0
            )
    
            ##########################
            # Temperature
            ##########################
    
            gpu.temperature_celsius = (
                pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            )
    
            ##########################
            # PCIe
            ##########################
    
            try:
                tx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle,
                    pynvml.NVML_PCIE_UTIL_TX_BYTES,
                )
    
                rx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle,
                    pynvml.NVML_PCIE_UTIL_RX_BYTES,
                )
    
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