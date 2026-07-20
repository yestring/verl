# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AMR Executor: Probe-then-Commit for configuration changes that require restart.

This module provides a lightweight mechanism to:
1. Save current checkpoint
2. Spawn a separate Ray task with modified config (e.g., TP/PP)
3. Run a few demo steps
4. Evaluate performance (throughput, loss stability)
5. Commit (keep new config) or rollback (restore checkpoint and continue with old config)
"""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

import ray
import torch.distributed as dist

from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.logger import logger

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
def run_demo_task(
    config_overrides: Dict[str, Any],
    checkpoint_dir: str,
    demo_steps: int,
    trainer_class,
    trainer_config,
):
    """
    Ray remote function to run a short demo with a modified configuration.

    Args:
        config_overrides: Dict of config paths to values (e.g., {"actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 4})
        checkpoint_dir: Path to the saved checkpoint to load.
        demo_steps: Number of training steps to run.
        trainer_class: The Trainer class (e.g., AthenaTrainer) to instantiate.
        trainer_config: The config object for the trainer.

    Returns:
        Dict with keys: "throughput", "mfu", "loss", "success" (bool)
    """
    # 1. Apply config overrides (modify in-memory config)
    # Since we are in a separate process, we can safely mutate the config.
    for path, value in config_overrides.items():
        keys = path.split(".")
        target = trainer_config
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

    # 2. Instantiate trainer and load checkpoint
    trainer = trainer_class(trainer_config)
    trainer.actor_engine.load_checkpoint(checkpoint_dir)

    # 3. Run demo_steps (we need to execute the training loop for a few steps)
    # We'll directly call the training step loop for demo_steps iterations.
    # This is a simplified version; actual implementation depends on VeRL's internals.
    throughputs = []
    losses = []
    mfus = []

    for step in range(demo_steps):
        # This is a placeholder; in practice you need to call the appropriate
        # method that runs one PPO iteration (rollout + update).
        # For simplicity, we assume a method `train_one_step()` exists.
        # You should replace this with the actual VeRL training step call.
        metrics = trainer.train_one_step()  # This must return throughput, loss, mfu
        throughputs.append(metrics.get("throughput", 0.0))
        losses.append(metrics.get("loss", 0.0))
        mfus.append(metrics.get("mfu", 0.0))

    avg_throughput = sum(throughputs) / len(throughputs)
    avg_loss = sum(losses) / len(losses)
    avg_mfu = sum(mfus) / len(mfus)

    return {
        "throughput": avg_throughput,
        "loss": avg_loss,
        "mfu": avg_mfu,
        "success": True,
    }


class AMRExecutor:
    """
    Probe-then-Commit executor for configuration changes that require restart.
    Uses Ray to run demo tasks without blocking the main training loop.
    """

    def __init__(self, trainer, max_attempts: int = 4):
        """
        Args:
            trainer: The main AthenaTrainer instance (must have actor_engine and checkpoint_manager).
            max_attempts: Maximum number of probe attempts before giving up.
        """
        self.trainer = trainer
        self.max_attempts = max_attempts
        self.attempt_count = 0
        self.consecutive_failures = 0
        self.baseline_throughput = None
        self.baseline_loss = None

        # We need a checkpoint manager to save/restore
        self.ckpt_manager = trainer.actor_engine.checkpoint_mananager

    def should_probe(self) -> bool:
        """Check if we should attempt a new probe."""
        if self.attempt_count >= self.max_attempts:
            logger.info(f"[AMRExecutor] Max attempts ({self.max_attempts}) reached, stopping probes.")
            return False
        if self.consecutive_failures >= 2:
            logger.info("[AMRExecutor] Two consecutive failures, early stopping.")
            return False
        return True

    def probe(
        self,
        candidate_config: Dict[str, Any],
        demo_steps: int,
        baseline_metrics: Dict[str, float],
    ) -> bool:
        """
        Execute a single probe: save checkpoint, run demo, evaluate, and commit/rollback.

        Args:
            candidate_config: Config overrides (e.g., {"actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 4})
            demo_steps: Number of steps to run in the demo.
            baseline_metrics: Dict with "throughput" and "loss" from the main training.

        Returns:
            bool: True if the probe was committed (success), False if rolled back.
        """
        self.attempt_count += 1
        self.baseline_throughput = baseline_metrics.get("throughput", 0.0)
        self.baseline_loss = baseline_metrics.get("loss", 0.0)

        # 1. Save current checkpoint
        tmp_ckpt_dir = tempfile.mkdtemp(prefix="athena_probe_")
        logger.info(f"[AMRExecutor] Saving checkpoint to {tmp_ckpt_dir}")
        self.ckpt_manager.save_checkpoint(
            local_path=tmp_ckpt_dir,
            global_step=self.trainer.global_steps,
        )
        dist.barrier()  # Ensure all ranks have saved

        # 2. Launch Ray task
        logger.info(f"[AMRExecutor] Launching probe task with config: {candidate_config}")
        future = run_demo_task.remote(
            config_overrides=candidate_config,
            checkpoint_dir=tmp_ckpt_dir,
            demo_steps=demo_steps,
            trainer_class=self.trainer.__class__,
            trainer_config=self.trainer.config,
        )

        try:
            result = ray.get(future, timeout=300)  # 5 min timeout
        except Exception as e:
            logger.error(f"[AMRExecutor] Probe task failed: {e}")
            self._rollback()
            return False

        # 3. Evaluate results
        probe_throughput = result.get("throughput", 0.0)
        probe_loss = result.get("loss", 0.0)
        probe_mfu = result.get("mfu", 0.0)

        logger.info(
            f"[AMRExecutor] Probe results: throughput={probe_throughput:.2f} (baseline={self.baseline_throughput:.2f}), "
            f"loss={probe_loss:.4f} (baseline={self.baseline_loss:.4f}), mfu={probe_mfu:.4f}"
        )

        # Accept if throughput improves > 3% and loss doesn't spike > 5%
        throughput_gain = (probe_throughput - self.baseline_throughput) / (self.baseline_throughput + 1e-8)
        loss_spike = (probe_loss - self.baseline_loss) / (self.baseline_loss + 1e-8)

        if throughput_gain > 0.03 and loss_spike < 0.05:
            logger.info("[AMRExecutor] Probe successful! Committing new configuration.")
            self._commit_config(candidate_config)
            self.consecutive_failures = 0
            return True
        else:
            logger.info("[AMRExecutor] Probe not beneficial, rolling back.")
            self._rollback()
            self.consecutive_failures += 1
            return False

    def _commit_config(self, config_overrides: Dict[str, Any]):
        """
        Apply the config overrides to the main trainer's config permanently.
        This usually involves modifying the trainer's config object and
        optionally updating the engine's internal state if possible.
        """
        for path, value in config_overrides.items():
            keys = path.split(".")
            target = self.trainer.config
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
        logger.info(f"[AMRExecutor] Committed config overrides: {config_overrides}")
        # Note: For TP/PP changes, the new config will be used when the trainer
        # restarts (e.g., after a checkpoint reload). For now, we just store it.

    def _rollback(self):
        """Rollback to the saved checkpoint (the probe's checkpoint is not committed)."""
        # Nothing to do here because the main trainer never changed its state.
        # The probe was run in a separate process.
        logger.info("[AMRExecutor] Rollback: no changes made to main trainer.")
        # We could optionally delete the temporary checkpoint directory.
        pass


def generate_candidate_config_for_tp_pp(
    current_tp: int,
    current_pp: int,
    strategy: str = "increase_tp",
) -> Dict[str, Any]:
    """
    Generate a candidate config override for TP/PP changes.

    Args:
        current_tp: Current tensor parallel size.
        current_pp: Current pipeline parallel size.
        strategy: "increase_tp", "increase_pp", "balance" (swap).

    Returns:
        Dict of config paths to new values.
    """
    overrides = {}
    if strategy == "increase_tp" and current_tp < 8:
        new_tp = current_tp * 2
        overrides["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"] = new_tp
        overrides["actor_rollout_ref.ref.megatron.tensor_model_parallel_size"] = new_tp
        overrides["critic.megatron.tensor_model_parallel_size"] = new_tp
        overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"] = new_tp
    elif strategy == "increase_pp" and current_pp < 4:
        new_pp = current_pp * 2
        overrides["actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"] = new_pp
        overrides["actor_rollout_ref.ref.megatron.pipeline_model_parallel_size"] = new_pp
        overrides["critic.megatron.pipeline_model_parallel_size"] = new_pp
    elif strategy == "balance":
        # Swap TP and PP if possible (e.g., TP=4,PP=2 -> TP=2,PP=4)
        if current_tp > 1 and current_pp > 1:
            overrides["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"] = current_pp
            overrides["actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"] = current_tp
            overrides["actor_rollout_ref.ref.megatron.tensor_model_parallel_size"] = current_pp
            overrides["actor_rollout_ref.ref.megatron.pipeline_model_parallel_size"] = current_tp
            overrides["critic.megatron.tensor_model_parallel_size"] = current_pp
            overrides["critic.megatron.pipeline_model_parallel_size"] = current_tp
            overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"] = current_pp
    return overrides