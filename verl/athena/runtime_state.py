# verl/athena/runtime_state.py

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUStats:
    """GPU 硬件及驱动层状态"""
    utilization: float = 0.0

    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    memory_peak_gb: float = 0.0
    memory_used_gb: float = 0.0

    memory_utilization: float = 0.0

    power_watts: float = 0.0
    temperature_celsius: float = 0.0

    pcie_tx_gb: float = 0.0
    pcie_rx_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if hasattr(v, 'to_dict'):
                result[k] = v.to_dict()
            elif isinstance(v, dict):
                result[k] = v
            else:
                result[k] = v
        return result


@dataclass
class VLLMStats:
    """vLLM 推理引擎内部状态"""
    num_running: int = 0                  # 当前正在处理的请求数
    num_waiting: int = 0                  # 等待队列中的请求数
    num_preempted: int = 0                # 累计被抢占的请求数
    num_evictions: int = 0                # KV Cache 被驱逐次数
    kv_usage: float = 0.0                 # KV Cache 使用率 (%)
    prefix_cache_hit: float = 0.0         # 前缀缓存命中率 (%)
    batch_size: int = 0                   # 当前推理批次大小
    prefill_tokens: int = 0               # Prefill 阶段处理的 token 数
    decode_tokens: int = 0                # Decode 阶段处理的 token 数
    scheduler_queue: int = 0              # 调度器队列长度

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TransferQueueStats:
    """TransferQueue 数据流转状态"""
    running: int = 0                      # 正在处理的任务数
    finished: int = 0                     # 已完成的任务数
    failure: int = 0                      # 失败的任务数
    pending: int = 0                      # 待处理任务数
    queue_length: int = 0                 # 队列当前长度
    latency_ms: float = 0.0               # 任务平均处理延迟 (ms)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RolloutStats:
    """Rollout 生成过程性能指标 (来自 AgentLoop)"""
    generate_latency_ms: float = 0.0      # 生成阶段总耗时 (ms)
    request_latency_ms: float = 0.0       # 单次请求端到端延迟 (ms)
    ttft_ms: float = 0.0                  # Time to First Token (首 token 延迟, ms)
    tpot_ms: float = 0.0                  # Time per Output Token (每 token 生成耗时, ms)
    num_preempted: int = 0                # 被抢占的请求数
    num_aborted: int = 0                  # 被中止的请求数

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingStats:
    """PPO 训练过程中的算法与损失指标"""
    entropy: float = 0.0                  # 策略熵
    grad_norm: float = 0.0                # 梯度范数
    lr: float = 0.0                       # 当前学习率
    ppo_kl: float = 0.0                   # PPO KL 散度 (近似)
    clip_fraction: float = 0.0            # 裁剪比例 (被 clip 的样本占比)
    pg_loss: float = 0.0                  # 策略梯度损失 (Policy Gradient loss)
    kl_loss: float = 0.0                  # KL 惩罚损失
    actor_loss: float = 0.0               # Actor 总损失
    critic_loss: float = 0.0              # Critic 损失 (value loss)
    reward: float = 0.0                   # 平均奖励
    advantage: float = 0.0                # 平均优势值
    returns: float = 0.0                  # 平均回报 (return)
    mfu: float = 0.0                      # Model FLOPs Utilization (模型算力利用率)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RuntimeState:
    """
    Athena 运行时状态顶层结构。
    聚合所有观测维度的数据，用于后续分析、可视化与决策。
    """
    # ---- 基础信息 ----
    global_step: int = 0
    epoch: int = 0
    timestamp: float = 0.0                # Unix 时间戳 (秒)

    # ---- 分阶段耗时 (来自 VeRL timing_raw) ----
    gen_time: float = 0.0                 # Rollout 生成耗时 (s)
    ref_time: float = 0.0                 # Reference model 前向耗时 (s)
    values_time: float = 0.0              # Critic value 计算耗时 (s)
    adv_time: float = 0.0                 # Advantage 计算耗时 (s)
    update_actor_time: float = 0.0        # Actor 更新耗时 (s)
    update_critic_time: float = 0.0       # Critic 更新耗时 (s)
    update_weights_time: float = 0.0      # 权重同步到 vLLM 耗时 (s)
    total_step_time: float = 0.0          # 整个 step 总耗时 (s)

    # ---- 吞吐与数据量 ----
    throughput: float = 0.0               # tokens/s (整体吞吐)
    total_num_tokens: int = 0             # 本 step 处理的总 token 数

    # ---- 序列长度 ----
    avg_prompt_len: float = 0.0
    avg_response_len: float = 0.0

    # ---- 各子模块状态 ----
    gpu: GPUStats = field(default_factory=GPUStats)
    vllm: VLLMStats = field(default_factory=VLLMStats)
    transfer_queue: TransferQueueStats = field(default_factory=TransferQueueStats)
    rollout: RolloutStats = field(default_factory=RolloutStats)
    training: TrainingStats = field(default_factory=TrainingStats)

    # ---- 额外扩展字段 (用于临时或未分类的数据) ----
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """递归转换为字典，便于 JSON 序列化"""
        result = {}
        for k, v in self.__dict__.items():
            if hasattr(v, 'to_dict'):
                result[k] = v.to_dict()
            elif isinstance(v, dict):
                result[k] = v
            else:
                result[k] = v
        return result

    def to_json(self) -> str:
        """输出 JSON 字符串 (使用 default=str 处理不可序列化对象)"""
        return json.dumps(self.to_dict(), default=str)