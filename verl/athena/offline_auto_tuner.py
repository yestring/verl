"""Restart-based black-box configuration profiler for Athena.

Every trial is a fresh VeRL subprocess and, when supplied, resumes from the
same checkpoint. Metrics are read from Athena's structured JSONL telemetry,
not scraped from console formatting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from verl.athena.parameters import get_by_path, set_by_path, validate_overrides
from verl.athena.utility import DEFAULT_WEIGHTS, UtilityScorer, UtilitySummary

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    trial: int
    config: dict[str, Any]
    status: str
    throughput: float = 0.0
    throughput_p10: float = 0.0
    throughput_cv: float = 0.0
    step_time: float = 0.0
    step_time_p95: float = 0.0
    gen_time: float = 0.0
    actor_time: float = 0.0
    peak_memory_gb: float = 0.0
    peak_reserved_memory_gb: float | None = None
    device_memory_used_gb: float | None = None
    device_memory_utilization: float | None = None
    gpu_utilization: float | None = None
    power_watts: float | None = None
    throughput_per_gb: float | None = None
    energy_per_1k_tokens: float | None = None
    mfu: float | None = None
    grad_norm: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    kv_usage: float | None = None
    transfer_queue_latency_ms: float | None = None
    samples: int = 0
    duration_s: float = 0.0
    gpu_count: int = 0
    utility: float | None = None
    utility_components: dict[str, Any] | None = None
    constraint_violations: list[str] | None = None
    metric_reliability: dict[str, str] | None = None
    error: str | None = None

    @property
    def feasible(self) -> bool:
        return self.status == "success" and not self.constraint_violations


def _checkpoint_step(checkpoint: Path | None) -> int:
    if checkpoint is None:
        return 0
    marker = "global_step_"
    if marker not in checkpoint.name:
        raise ValueError(f"Checkpoint path must end in global_step_N: {checkpoint}")
    return int(checkpoint.name.rsplit(marker, 1)[1])


def load_probe_states(log_dir: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("trace_step_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                states.append(json.loads(line))
    states.sort(key=lambda item: int(item.get("global_step", 0)))
    return states


def load_restart_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Load cold candidates emitted by a running Athena trainer."""
    candidates = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        request = json.loads(line)
        candidate = request.get("candidate_config")
        if candidate:
            candidates.append(candidate)
    return candidates


def load_console_metric(path: Path, metric_name: str, warmup_steps: int = 0) -> list[float]:
    """Best-effort recovery for metrics omitted from legacy Athena JSONL."""
    if not path.exists():
        return []
    pattern = re.compile(rf"{re.escape(metric_name)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    values = [float(match.group(1)) for match in pattern.finditer(path.read_text())]
    return values[warmup_steps:]


def summarize_probe(states: list[dict[str, Any]], warmup_steps: int) -> dict[str, Any]:
    measured = states[warmup_steps:]
    if not measured:
        raise RuntimeError(f"Probe produced {len(states)} states; none remain after {warmup_steps} warmup steps")

    def values(path: tuple[str, ...]) -> list[float]:
        output = []
        for state in measured:
            value: Any = state
            for key in path:
                value = value.get(key, 0.0) if isinstance(value, dict) else 0.0
            output.append(float(value or 0.0))
        return output

    throughput_values = values(("throughput",))
    step_time_values = values(("total_step_time",))

    def percentile(items: list[float], fraction: float) -> float:
        ordered = sorted(items)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    mean_throughput = statistics.fmean(throughput_values)
    peak_memory = max(values(("gpu", "memory_peak_gb")))
    reliability: dict[str, str] = {
        "throughput": "reliable",
        "step_time": "reliable",
        "peak_memory_gb": "reliable" if peak_memory > 0 else "all_zero",
    }

    def optional_metric(
        path: tuple[str, ...],
        source_name: str,
        aggregate: Callable[[list[float]], float] = statistics.median,
        snapshot: bool = False,
    ) -> float | None:
        metric_values = values(path)
        sources = [state.get("extra", {}).get("metric_sources", {}).get(source_name) for state in measured]
        declared_sources = [source for source in sources if source is not None]
        if declared_sources and all(source == "unavailable" for source in declared_sources):
            reliability[source_name] = "unavailable"
            return None
        if all(value == 0 for value in metric_values):
            reliability[source_name] = "all_zero"
            return None
        if snapshot or any("snapshot" in source for source in declared_sources):
            reliability[source_name] = "partial"
        elif declared_sources:
            reliability[source_name] = "reliable"
        else:
            reliability[source_name] = "legacy_unverified"
        return aggregate(metric_values)

    reserved_memory = optional_metric(("gpu", "memory_reserved_gb"), "gpu.memory_reserved_gb", aggregate=max)
    device_memory = optional_metric(("gpu", "memory_used_gb"), "gpu.memory_used_gb", aggregate=max, snapshot=True)
    device_memory_utilization = optional_metric(
        ("gpu", "memory_utilization"), "gpu.memory_utilization", aggregate=max, snapshot=True
    )
    gpu_utilization = optional_metric(
        ("gpu", "utilization"), "gpu.utilization", aggregate=statistics.fmean, snapshot=True
    )
    mean_power = optional_metric(("gpu", "power_watts"), "gpu.power_watts", aggregate=statistics.fmean, snapshot=True)
    mfu = optional_metric(("training", "mfu"), "training.mfu")
    grad_norm = optional_metric(("training", "grad_norm"), "training.grad_norm")
    ttft = optional_metric(("rollout", "ttft_ms"), "rollout.ttft_ms")
    tpot = optional_metric(("rollout", "tpot_ms"), "rollout.tpot_ms")
    kv_usage = optional_metric(("vllm", "kv_usage"), "vllm.kv_usage")
    tq_latency = optional_metric(
        ("transfer_queue", "latency_ms"), "transfer_queue.latency_ms", aggregate=statistics.fmean
    )

    throughput_per_gb = statistics.median(throughput_values) / peak_memory if peak_memory else None
    energy_per_1k_tokens = 1000.0 * mean_power / mean_throughput if mean_power is not None and mean_throughput else None
    reliability["throughput_per_gb"] = "derived" if throughput_per_gb is not None else "unavailable"
    reliability["energy_per_1k_tokens"] = "derived_from_partial" if energy_per_1k_tokens is not None else "unavailable"
    return {
        "throughput": statistics.median(throughput_values),
        "throughput_p10": percentile(throughput_values, 0.10),
        "throughput_cv": statistics.pstdev(throughput_values) / mean_throughput if mean_throughput else 0.0,
        "step_time": statistics.median(step_time_values),
        "step_time_p95": percentile(step_time_values, 0.95),
        "gen_time": statistics.median(values(("gen_time",))),
        "actor_time": statistics.median(values(("update_actor_time",))),
        "peak_memory_gb": peak_memory,
        "peak_reserved_memory_gb": reserved_memory,
        "device_memory_used_gb": device_memory,
        "device_memory_utilization": device_memory_utilization,
        "gpu_utilization": gpu_utilization,
        "power_watts": mean_power,
        "throughput_per_gb": throughput_per_gb,
        "energy_per_1k_tokens": energy_per_1k_tokens,
        "mfu": mfu,
        "grad_norm": grad_norm,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "kv_usage": kv_usage,
        "transfer_queue_latency_ms": tq_latency,
        "samples": len(measured),
        "metric_reliability": reliability,
    }


class OfflineAutoTuner:
    """Run isolated probes and select the fastest memory-safe configuration."""

    def __init__(
        self,
        base_config_path: str | Path,
        probe_steps: int = 20,
        warmup_steps: int = 5,
        work_dir: str | Path = "./athena_tuning",
        resume_from: str | Path | None = None,
        memory_limit_gb: float | None = None,
        timeout_s: int = 1800,
        utility_weights: Mapping[str, float] | None = None,
        constraints: Mapping[str, Mapping[str, float]] | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.base_config_path = Path(base_config_path).resolve()
        self.probe_steps = probe_steps
        self.warmup_steps = warmup_steps
        self.work_dir = Path(work_dir).resolve()
        self.resume_from = Path(resume_from).resolve() if resume_from else None
        self.timeout_s = timeout_s
        self.command_runner = command_runner
        self.repo_root = Path(__file__).resolve().parents[2]
        self.base_config = self._load_config()
        model_engine = str(self.base_config.get("model_engine", "dp"))
        self.backend = "fsdp" if model_engine == "dp" else model_engine
        self.gpu_count = int(get_by_path(self.base_config, "trainer.n_gpus_per_node", 1)) * int(
            get_by_path(self.base_config, "trainer.nnodes", 1)
        )
        offline_config = get_by_path(self.base_config, "trainer.athena.offline", {}) or {}
        configured_weights = offline_config.get("utility_weights", None)
        configured_constraints = deepcopy(offline_config.get("constraints", {}) or {})
        if constraints:
            for metric, bounds in constraints.items():
                configured_constraints.setdefault(metric, {}).update(bounds)
        # Keep the original CLI/API memory limit as a backward-compatible
        # shorthand for the generic peak-memory constraint.
        memory_bounds = configured_constraints.setdefault("peak_memory_gb", {})
        if memory_limit_gb is not None:
            memory_bounds["max"] = float(memory_limit_gb)
        else:
            memory_bounds.setdefault("max", 38.0)
        self.memory_limit_gb = float(memory_bounds["max"])
        self.utility_scorer = UtilityScorer(
            weights=utility_weights or configured_weights or DEFAULT_WEIGHTS,
            constraints=configured_constraints,
        )
        self.utility_summary: UtilitySummary | None = None
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.resume_from is not None and not self.resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {self.resume_from}")

    def _load_config(self) -> dict[str, Any]:
        config_dir = Path(__file__).parents[1] / "trainer" / "config"
        user_config = OmegaConf.load(self.base_config_path)
        requested_engine = str(user_config.get("model_engine", "dp"))
        # ``OmegaConf.load(ppo_trainer.yaml)`` is insufficient because VeRL's
        # component configs are selected through Hydra's defaults list.
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
            default_config = compose(config_name="ppo_trainer", overrides=[f"model_engine={requested_engine}"])
        OmegaConf.set_struct(default_config, False)
        user_config.pop("defaults", None)
        merged = OmegaConf.merge(default_config, user_config)
        # Catch mandatory fields on active paths before spending minutes
        # starting Ray. Disabled optional components legitimately retain ???.
        for path in ("actor_rollout_ref.model.path", "actor_rollout_ref.rollout.name"):
            try:
                OmegaConf.select(merged, path, throw_on_missing=True)
            except Exception as exc:
                raise ValueError(f"Athena base config must set {path}") from exc
        return OmegaConf.to_container(merged, resolve=True)

    def _trial_config(self, overrides: Mapping[str, Any], trial_dir: Path) -> dict[str, Any]:
        validate_overrides(overrides, backend=self.backend)
        config = deepcopy(self.base_config)
        for path, value in overrides.items():
            set_by_path(config, path, value)
        rollout_n = overrides.get("actor_rollout_ref.rollout.n")
        if rollout_n is not None:
            # These fields originate from OmegaConf interpolation, but the
            # tuner intentionally resolves the base config before applying
            # trial overrides. Keep every consumer aligned explicitly.
            for role in ("actor", "ref", "critic"):
                role_path = f"actor_rollout_ref.{role}"
                if get_by_path(config, role_path, None) is not None:
                    set_by_path(config, f"{role_path}.rollout_n", rollout_n)
        if self.backend == "megatron":
            tp = overrides.get("actor_rollout_ref.actor.megatron.tensor_model_parallel_size")
            pp = overrides.get("actor_rollout_ref.actor.megatron.pipeline_model_parallel_size")
            if tp is not None and pp is not None:
                # The resolved base config no longer retains OmegaConf
                # interpolation from actor to colocated ref/critic.
                for role in ("ref", "critic"):
                    prefix = f"actor_rollout_ref.{role}"
                    if get_by_path(config, f"{prefix}.strategy", None) == "megatron":
                        set_by_path(config, f"{prefix}.megatron.tensor_model_parallel_size", tp)
                        set_by_path(config, f"{prefix}.megatron.pipeline_model_parallel_size", pp)

        start_step = _checkpoint_step(self.resume_from)
        trainer = config.setdefault("trainer", {})
        trainer["total_training_steps"] = start_step + self.warmup_steps + self.probe_steps
        trainer["save_freq"] = -1
        trainer["test_freq"] = -1
        trainer["val_before_train"] = False
        trainer["logger"] = ["console"]
        trainer["default_local_dir"] = str(trial_dir / "checkpoints")
        trainer["athena_log_dir"] = str(trial_dir / "telemetry")
        trainer["resume_mode"] = "resume_path" if self.resume_from else "disable"
        trainer["resume_from_path"] = str(self.resume_from) if self.resume_from else None
        # Probes observe only. They must not recursively request new probes or
        # change the token budget halfway through a trial.
        trainer.setdefault("athena", {})["enabled"] = False
        return config

    def run_probe(self, trial: int, overrides: Mapping[str, Any]) -> ProbeResult:
        trial_dir = self.work_dir / f"trial_{trial:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = self._trial_config(overrides, trial_dir)
        config_path = trial_dir / "probe.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "--config-path",
            str(trial_dir),
            "--config-name",
            "probe",
        ]
        env = os.environ.copy()
        env.setdefault("VERL_DISABLE_WANDB", "1")
        started = time.monotonic()
        try:
            completed = self.command_runner(
                command,
                cwd=str(self.repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return ProbeResult(
                trial,
                dict(overrides),
                "timeout",
                duration_s=time.monotonic() - started,
                gpu_count=self.gpu_count,
                error=str(exc),
            )

        duration = time.monotonic() - started
        (trial_dir / "stdout.log").write_text(completed.stdout or "")
        (trial_dir / "stderr.log").write_text(completed.stderr or "")
        if completed.returncode != 0:
            output = f"{completed.stdout}\n{completed.stderr}"
            status = "oom" if "out of memory" in output.lower() else "failed"
            return ProbeResult(
                trial, dict(overrides), status, duration_s=duration, gpu_count=self.gpu_count, error=output[-4000:]
            )

        try:
            summary = summarize_probe(load_probe_states(trial_dir / "telemetry"), self.warmup_steps)
        except Exception as exc:
            return ProbeResult(
                trial,
                dict(overrides),
                "invalid_metrics",
                duration_s=duration,
                gpu_count=self.gpu_count,
                error=str(exc),
            )
        return ProbeResult(trial, dict(overrides), "success", duration_s=duration, gpu_count=self.gpu_count, **summary)

    def _select_best(self, results: list[ProbeResult]) -> ProbeResult | None:
        self.utility_summary = self.utility_scorer.score(results)
        feasible = [result for result in results if result.feasible]
        return max(feasible, key=lambda result: result.utility or 0.0) if feasible else None

    def search(self, search_space: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], list[ProbeResult]]:
        results = [self.run_probe(index, candidate) for index, candidate in enumerate(search_space)]
        best = self._select_best(results)
        if best is None:
            self.write_report(results, None)
            raise RuntimeError("No probe satisfied all constraints; see tuning_report.json")
        self.write_report(results, best)
        return best.config, results

    def search_optuna(self, n_trials: int) -> tuple[dict[str, Any], list[ProbeResult]]:
        """Use Optuna's TPE sampler while retaining the same probe executor.

        Optuna is an optional dependency so the deterministic grid workflow
        remains usable on restricted clusters.
        """
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Optuna search requires `uv pip install optuna`") from exc

        results: list[ProbeResult] = []

        def objective(trial) -> float:
            train_batch_size = trial.suggest_categorical("train_batch_size", [4, 8, 16, 32])
            ppo_mini_batch_size = trial.suggest_categorical("ppo_mini_batch_size", [1, 2, 4])
            if ppo_mini_batch_size > train_batch_size:
                raise optuna.TrialPruned(
                    f"ppo_mini_batch_size={ppo_mini_batch_size} exceeds train_batch_size={train_batch_size}"
                )
            overrides = {
                "data.train_batch_size": train_batch_size,
                "actor_rollout_ref.actor.ppo_mini_batch_size": ppo_mini_batch_size,
                "actor_rollout_ref.actor.use_dynamic_bsz": True,
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": trial.suggest_categorical(
                    "token_budget", [2048, 4096, 8192, 12288]
                ),
                "actor_rollout_ref.model.enable_gradient_checkpointing": trial.suggest_categorical(
                    "gradient_checkpointing", [True, False]
                ),
                "actor_rollout_ref.rollout.n": trial.suggest_categorical("rollout_n", [2, 4, 8]),
                "actor_rollout_ref.rollout.gpu_memory_utilization": trial.suggest_categorical(
                    "rollout_gpu_memory_utilization", [0.2,0.3,0.4,0.5]
                ),
                "actor_rollout_ref.rollout.max_num_seqs": trial.suggest_categorical(
                    "rollout_max_num_seqs", [8, 16, 32]
                ),
                "data.dataloader_num_workers": trial.suggest_categorical("dataloader_num_workers", [2, 4, 8]),
                "data.max_prompt_length": trial.suggest_categorical("max_prompt_length", [128, 256, 512]),
                "data.max_response_length": trial.suggest_categorical("max_response_length", [64, 128, 256]),
            }
            if self.backend == "megatron":
                tp_choices = [value for value in (1, 2) if value <= self.gpu_count]
                pp_choices = [value for value in (1, 2) if value <= self.gpu_count]
                tp = trial.suggest_categorical("tensor_model_parallel_size", tp_choices)
                pp = trial.suggest_categorical("pipeline_model_parallel_size", pp_choices)

                if tp==2 and pp==2:
                    pp=1
                
                if tp * pp > self.gpu_count:
                    raise optuna.TrialPruned(f"TP*PP={tp * pp} exceeds gpu_count={self.gpu_count}")
                overrides.update(
                    {
                        "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": tp,
                        "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": pp,
                    }
                )
            else:
                param_offload = trial.suggest_categorical("param_offload", [False, True])
                overrides.update(
                    {
                        "actor_rollout_ref.actor.fsdp_config.sharding_strategy": trial.suggest_categorical(
                            "sharding_strategy", ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]
                        ),
                        "actor_rollout_ref.actor.fsdp_config.param_offload": param_offload,
                        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": trial.suggest_categorical(
                            "optimizer_offload", [False, True]
                        )
                        if param_offload
                        else False,
                    }
                )
            # ---- entropy kernel decision space (restart-only, probed offline) ----
            use_fused_kernels = trial.suggest_categorical("use_fused_kernels", [False, True])
            kernel_prefix = (
                "actor_rollout_ref.actor.megatron"
                if self.backend == "megatron"
                else "actor_rollout_ref.actor.fsdp_config"
            )
            kernel_overrides = {
                "actor_rollout_ref.model.use_fused_kernels": use_fused_kernels,
            }
            if use_fused_kernels:
                # Lazy import: the tuner may run on a CPU-only head node and
                # kernels.py is safe to import without triton/CUDA.
                from verl.utils.kernel.kernels import SUPPORTED_BACKWARD_METHODS

                kernel_overrides.update(
                    {
                        f"{kernel_prefix}.entropy_kernel_backward_method": trial.suggest_categorical(
                            "entropy_kernel_backward_method", list(SUPPORTED_BACKWARD_METHODS)
                        ),
                        f"{kernel_prefix}.entropy_kernel_forward_fused": trial.suggest_categorical(
                            "entropy_kernel_forward_fused", [False, True]
                        ),
                        f"{kernel_prefix}.entropy_kernel_vocab_per_split": trial.suggest_categorical(
                            "entropy_kernel_vocab_per_split", [4096, 8192, 16384, 32768]
                        ),
                        f"{kernel_prefix}.entropy_kernel_n_group": trial.suggest_categorical(
                            "entropy_kernel_n_group", [2, 4, 8, 16]
                        ),
                    }
                )
            overrides.update(kernel_overrides)
            try:
                result = self.run_probe(trial.number, overrides)
            except Exception as exc:
                logger.exception("Optuna trial %s failed before probe completion", trial.number)
                result = ProbeResult(
                    trial=trial.number,
                    config=dict(overrides),
                    status="failed",
                    gpu_count=self.gpu_count,
                    error=str(exc),
                )
            results.append(result)
            trial.set_user_attr("probe_result", asdict(result))
            self.utility_scorer.apply_constraints([result])
            if not result.feasible:
                raise optuna.TrialPruned(result.status)
            return result.throughput

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=None))
        study.optimize(objective, n_trials=n_trials)
        best = self._select_best(results)
        if best is None:
            self.write_report(results, None)
            raise RuntimeError("No Optuna probe satisfied all constraints")
        self.write_report(results, best)
        return best.config, results

    def apply_best_config(self, overrides: Mapping[str, Any], output_path: str | Path | None = None) -> Path:
        config = deepcopy(self.base_config)
        for path, value in overrides.items():
            set_by_path(config, path, value)
        output = Path(output_path).resolve() if output_path else self.work_dir / "best_config.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(config, sort_keys=False))
        return output

    def write_report(self, results: list[ProbeResult], best: ProbeResult | None) -> Path:
        report = {
            "base_config": str(self.base_config_path),
            "resume_from": str(self.resume_from) if self.resume_from else None,
            "warmup_steps": self.warmup_steps,
            "probe_steps": self.probe_steps,
            "memory_limit_gb": self.memory_limit_gb,
            "utility": asdict(self.utility_summary) if self.utility_summary else None,
            "best_trial": best.trial if best else None,
            "best_config": best.config if best else None,
            "results": [asdict(result) for result in results],
        }
        path = self.work_dir / "tuning_report.json"
        path.write_text(json.dumps(report, indent=2))
        return path


def build_default_fsdp_search_space(base_config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """A small, semantically fair single-GPU search space.

    Sequence lengths and optimizer hyperparameters are intentionally excluded:
    changing them would alter the workload or learning objective.
    """
    dynamic_candidates = [
        {
            "actor_rollout_ref.actor.use_dynamic_bsz": True,
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": token_budget,
            "actor_rollout_ref.model.enable_gradient_checkpointing": checkpointing,
        }
        for token_budget in (2048, 4096, 8192)
        for checkpointing in (True, False)
    ]
    # Entropy kernel candidates: fixed fused baseline, sweep the backward
    # strategy, then one fused-forward and one tiling combination.
    dynamic_candidates.extend(
        {
            "actor_rollout_ref.model.use_fused_kernels": True,
            "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method": method,
        }
        for method in ("split_dlogits_n", "split_fused_n", "split_fused_n_grouped", "owner_compute")
    )
    dynamic_candidates.extend(
        [
            {
                "actor_rollout_ref.model.use_fused_kernels": True,
                "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method": "split_fused_n",
                "actor_rollout_ref.actor.fsdp_config.entropy_kernel_forward_fused": True,
            },
            {
                "actor_rollout_ref.model.use_fused_kernels": True,
                "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method": "split_fused_n_grouped",
                "actor_rollout_ref.actor.fsdp_config.entropy_kernel_vocab_per_split": 16384,
                "actor_rollout_ref.actor.fsdp_config.entropy_kernel_n_group": 8,
            },
        ]
    )
    if base_config is None:
        return dynamic_candidates

    # Offload is evaluated independently because it trades GPU memory for PCIe
    # traffic and is especially useful as a memory-safe fallback.
    current_budget = int(get_by_path(base_config, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", 4096))
    dynamic_candidates.extend(
        [
            {
                "actor_rollout_ref.actor.use_dynamic_bsz": True,
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": current_budget,
                "actor_rollout_ref.actor.fsdp_config.param_offload": param_offload,
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload": optimizer_offload,
            }
            for param_offload, optimizer_offload in ((False, False), (True, False), (True, True))
        ]
    )
    # Deduplicate candidates when the base token budget is already one of the
    # standard values.
    unique: dict[str, dict[str, Any]] = {}
    for candidate in dynamic_candidates:
        unique[json.dumps(candidate, sort_keys=True)] = candidate
    return list(unique.values())


def build_default_megatron_search_space(base_config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Conservative Megatron probes; TP/PP stay at one on a single GPU."""
    candidates = [
        {
            "actor_rollout_ref.actor.use_dynamic_bsz": True,
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": token_budget,
            "actor_rollout_ref.model.enable_gradient_checkpointing": checkpointing,
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 1,
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": 1,
        }
        for token_budget in (128, 256, 512)
        for checkpointing in (True, False)
    ]
    # Entropy kernel candidates (mirror the FSDP set with megatron paths).
    candidates.extend(
        {
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 1,
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": 1,
            "actor_rollout_ref.model.use_fused_kernels": True,
            "actor_rollout_ref.actor.megatron.entropy_kernel_backward_method": method,
        }
        for method in ("split_dlogits_n", "split_fused_n", "split_fused_n_grouped", "owner_compute")
    )
    candidates.extend(
        [
            {
                "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 1,
                "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": 1,
                "actor_rollout_ref.model.use_fused_kernels": True,
                "actor_rollout_ref.actor.megatron.entropy_kernel_backward_method": "split_fused_n",
                "actor_rollout_ref.actor.megatron.entropy_kernel_forward_fused": True,
            },
            {
                "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 1,
                "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": 1,
                "actor_rollout_ref.model.use_fused_kernels": True,
                "actor_rollout_ref.actor.megatron.entropy_kernel_backward_method": "split_fused_n_grouped",
                "actor_rollout_ref.actor.megatron.entropy_kernel_vocab_per_split": 16384,
                "actor_rollout_ref.actor.megatron.entropy_kernel_n_group": 8,
            },
        ]
    )
    return candidates


def parse_weight_arguments(arguments: Iterable[str]) -> dict[str, float] | None:
    weights = {}
    for argument in arguments:
        if "=" not in argument:
            raise ValueError(f"Weight must use METRIC=VALUE syntax: {argument}")
        metric, value = argument.split("=", 1)
        weights[metric.strip()] = float(value)
    return weights or None


def parse_constraint_arguments(arguments: Iterable[str]) -> dict[str, dict[str, float]]:
    constraints: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(<=|>=|==)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    operators = {"<=": "max", ">=": "min", "==": "eq"}
    for argument in arguments:
        match = pattern.fullmatch(argument.replace(" ", ""))
        if match is None:
            raise ValueError(f"Constraint must look like peak_memory_gb<=8: {argument}")
        metric, operator, value = match.groups()
        constraints.setdefault(metric, {})[operators[operator]] = float(value)
    return constraints


def rescore_report(
    report_path: str | Path,
    weights: Mapping[str, float] | None = None,
    constraints: Mapping[str, Mapping[str, float]] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Re-score completed probes without launching GPU work."""
    source = Path(report_path)
    payload = json.loads(source.read_text())
    inferred_gpu_count = 0
    base_config_path = payload.get("base_config")
    if base_config_path and Path(base_config_path).exists():
        raw_config = OmegaConf.to_container(OmegaConf.load(base_config_path), resolve=True)
        inferred_gpu_count = int(get_by_path(raw_config, "trainer.n_gpus_per_node", 1)) * int(
            get_by_path(raw_config, "trainer.nnodes", 1)
        )
    result_payloads = []
    warmup_steps = int(payload.get("warmup_steps", 0))
    for item in payload.get("results", []):
        item = dict(item)
        item.setdefault("gpu_count", inferred_gpu_count)
        telemetry_dir = source.parent / f"trial_{int(item['trial']):04d}" / "telemetry"
        if telemetry_dir.exists():
            try:
                summary = summarize_probe(load_probe_states(telemetry_dir), warmup_steps)
                reserved_values = load_console_metric(
                    telemetry_dir.parent / "stdout.log", "actor/perf/max_memory_reserved_gb", warmup_steps
                )
                if summary.get("peak_reserved_memory_gb") is None and reserved_values:
                    summary["peak_reserved_memory_gb"] = max(reserved_values)
                    summary["metric_reliability"]["gpu.memory_reserved_gb"] = "legacy_console_recovered"
                for metric, value in summary.items():
                    item[metric] = value
            except Exception:
                # Old reports remain scoreable even if their raw telemetry was
                # moved or partially deleted.
                pass
        result_payloads.append(item)
    results = [ProbeResult(**item) for item in result_payloads]
    effective_constraints = deepcopy(constraints or {})
    if payload.get("memory_limit_gb") is not None:
        effective_constraints.setdefault("peak_memory_gb", {}).setdefault("max", payload["memory_limit_gb"])
    scorer = UtilityScorer(weights=weights, constraints=effective_constraints)
    summary = scorer.score(results)
    feasible = [result for result in results if result.feasible]
    best = max(feasible, key=lambda result: result.utility or 0.0) if feasible else None
    payload.update(
        {
            "rescored_from": str(source.resolve()),
            "utility": asdict(summary),
            "best_trial": best.trial if best else None,
            "best_config": best.config if best else None,
            "results": [asdict(result) for result in results],
        }
    )
    output = Path(output_path) if output_path else source.with_name(f"{source.stem}_rescored.json")
    output.write_text(json.dumps(payload, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Athena restart-based offline tuning")
    parser.add_argument("--config", help="Base VeRL YAML config")
    parser.add_argument("--rescore-report", default=None, help="Re-score an existing report without running probes")
    parser.add_argument("--probe-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--work-dir", default="./athena_tuning")
    parser.add_argument("--resume-from", default=None, help="Optional global_step_N checkpoint shared by all trials")
    parser.add_argument(
        "--restart-requests",
        default=None,
        help="Optional restart_requests.jsonl emitted by AthenaTrainer; candidates are probed before defaults",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=None,
        help="Backward-compatible shorthand for --constraint peak_memory_gb<=VALUE.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Utility weight; repeat for multiple metrics. Overrides weights in trainer.athena.offline.",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        metavar="EXPR",
        help="Generic numeric constraint, e.g. peak_memory_gb<=8 or gpu_count==2.",
    )
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--sampler", choices=("grid", "optuna"), default="grid")
    parser.add_argument("--trials", type=int, default=12, help="Number of trials for --sampler=optuna")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    weights = parse_weight_arguments(args.weight)
    constraints = parse_constraint_arguments(args.constraint)
    if args.memory_limit_gb is not None:
        constraints.setdefault("peak_memory_gb", {})["max"] = args.memory_limit_gb
    if args.rescore_report:
        output = rescore_report(args.rescore_report, weights=weights, constraints=constraints, output_path=args.output)
        print(f"[Athena] Re-scored report written to {output}")
        return
    if not args.config:
        parser.error("--config is required unless --rescore-report is used")

    tuner = OfflineAutoTuner(
        args.config,
        probe_steps=args.probe_steps,
        warmup_steps=args.warmup_steps,
        work_dir=args.work_dir,
        resume_from=args.resume_from,
        memory_limit_gb=args.memory_limit_gb,
        timeout_s=args.timeout_s,
        utility_weights=weights,
        constraints=constraints,
    )
    if args.sampler == "optuna":
        best, _ = tuner.search_optuna(args.trials)
    else:
        search_space = load_restart_candidates(args.restart_requests) if args.restart_requests else []
        defaults = (
            build_default_megatron_search_space(tuner.base_config)
            if tuner.backend == "megatron"
            else build_default_fsdp_search_space(tuner.base_config)
        )
        search_space.extend(defaults)
        best, _ = tuner.search(search_space)
    output = tuner.apply_best_config(best, args.output)
    print(f"[Athena] Best configuration written to {output}")


if __name__ == "__main__":
    main()
