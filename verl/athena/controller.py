# verl/athena/controller.py
"""
Athena Dynamic Controller - 基于真实可控点的闭环优化

功能：
1. 权重同步自适应（Weight Sync Adaptive Control）
   - 根据 Off-Policy 陈旧度（staleness）决定是否跳过 update_weights
   - 减少同步开销，提升训练吞吐

2. Rollout 并发预留接口（暂不启用，供未来扩展）
3. Sampling Params 预留接口（暂不启用，需 AgentLoop 配合）
"""

import logging
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class WeightSyncConfig:
    """权重同步策略配置"""
    staleness_threshold_skip: float = 1.0      # staleness ≤ 此值可以跳过同步
    staleness_threshold_force: float = 3.0     # staleness ≥ 此值强制同步
    max_skip_steps: int = 5                    # 最大连续跳过步数
    cooldown_steps: int = 3                    # 同步后冷却步数


@dataclass
class WeightSyncState:
    """权重同步策略运行时状态"""
    consecutive_skips: int = 0                 # 当前连续跳过次数
    last_sync_step: int = 0                    # 上次同步的 step
    cooldown_remaining: int = 0                # 冷却剩余步数
    should_skip: bool = False                  # 当前决策：是否跳过同步


class AthenaController:
    """
    Athena 运行时控制器

    用法：
        1. 在 AthenaTrainer._compute_metrics() 中调用 controller.observe(state)
        2. 在 AthenaTrainer.on_step_end() 中调用 controller.should_sync()
        3. 根据返回值决定是否执行 checkpoint_manager.update_weights()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.weight_sync_config = WeightSyncConfig(
            **(self.config.get("weight_sync", {}))
        )
        self._ws_state = WeightSyncState()
        self._stats = {
            "sync_count": 0,
            "skip_count": 0,
            "last_decision": None,
            "last_staleness": 0.0,
        }

    # ============================================================
    # 主入口
    # ============================================================

    def observe(self, state: Dict[str, Any], global_step: int):
        """
        接收最新的 RuntimeState，执行所有策略的决策逻辑

        Args:
            state: RuntimeState 的字典形式（来自 runtime_state.to_dict()）
            global_step: 当前训练步数
        """
        self._decide_weight_sync(state, global_step)
        # 可扩展其他策略：_decide_rollout_concurrency(state, global_step) 等

    def should_sync(self) -> bool:
        """
        返回当前 step 是否应该执行权重同步

        Returns:
            True: 执行同步（默认行为）
            False: 跳过同步
        """
        return not self._ws_state.should_skip

    # ============================================================
    # 策略：权重同步自适应
    # ============================================================

    def _decide_weight_sync(self, state: Dict[str, Any], global_step: int):
        """
        基于 Off-Policy 陈旧度决定是否跳过权重同步
        """
        ws = self._ws_state
        cfg = self.weight_sync_config

        # 1. 提取 staleness（从 extra 中获取）
        extra = state.get("extra", {})
        staleness = extra.get("off_policy_staleness", 0.0)
        self._stats["last_staleness"] = staleness

        # 2. 冷却检查
        if ws.cooldown_remaining > 0:
            ws.cooldown_remaining -= 1
            ws.should_skip = False   # 冷却期强制同步
            self._stats["last_decision"] = "cooldown"
            return

        # 3. 强制同步条件
        if staleness >= cfg.staleness_threshold_force:
            ws.should_skip = False
            ws.consecutive_skips = 0
            ws.cooldown_remaining = cfg.cooldown_steps
            self._stats["last_decision"] = "force_sync"
            self._stats["sync_count"] += 1
            logger.info(f"[Athena] WeightSync: FORCE SYNC (staleness={staleness:.2f})")
            return

        # 4. 可以跳过的条件
        if (staleness <= cfg.staleness_threshold_skip and
            ws.consecutive_skips < cfg.max_skip_steps):
            ws.should_skip = True
            ws.consecutive_skips += 1
            self._stats["last_decision"] = "skip"
            self._stats["skip_count"] += 1
            logger.info(f"[Athena] WeightSync: SKIP (skip #{ws.consecutive_skips}, staleness={staleness:.2f})")
            return

        # 5. 默认：同步
        ws.should_skip = False
        ws.consecutive_skips = 0
        self._stats["last_decision"] = "sync"
        self._stats["sync_count"] += 1
        logger.info(f"[Athena] WeightSync: SYNC (staleness={staleness:.2f})")

    # ============================================================
    # 预留接口（供未来扩展）
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """返回控制器统计信息"""
        return {
            "weight_sync": {
                "sync_count": self._stats["sync_count"],
                "skip_count": self._stats["skip_count"],
                "last_decision": self._stats["last_decision"],
                "last_staleness": self._stats["last_staleness"],
                "consecutive_skips": self._ws_state.consecutive_skips,
                "cooldown_remaining": self._ws_state.cooldown_remaining,
            }
        }