"""Restart-based black-box configuration profiler for Athena.

Every trial is a fresh VeRL subprocess and, when supplied, resumes from the
same checkpoint. Metrics are read from Athena's structured JSONL telemetry,
not scraped from console formatting.
"""

from __future__ import annotations

import argparse
import json
import os
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


@dataclass
class ProbeResult:
    trial: int
    config: dict[str, Any]
    status: str
    throughput: float = 0.0
    step_time: float = 0.0
    peak_memory_gb: float = 0.0
    gpu_utilization: float = 0.0
    mfu: float = 0.0
    samples: int = 0
    duration_s: float = 0.0
    error: str | None = None

    @property
    def feasible(self) -> bool:
        return self.status == "success"


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


def summarize_probe(states: list[dict[str, Any]], warmup_steps: int) -> dict[str, float]:
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

    return {
        "throughput": statistics.median(values(("throughput",))),
        "step_time": statistics.median(values(("total_step_time",))),
        "peak_memory_gb": max(values(("gpu", "memory_peak_gb"))),
        "gpu_utilization": statistics.fmean(values(("gpu", "utilization"))),
        "mfu": statistics.median(values(("training", "mfu"))),
        "samples": len(measured),
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
        memory_limit_gb: float = 38.0,
        timeout_s: int = 1800,
        command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.base_config_path = Path(base_config_path).resolve()
        self.probe_steps = probe_steps
        self.warmup_steps = warmup_steps
        self.work_dir = Path(work_dir).resolve()
        self.resume_from = Path(resume_from).resolve() if resume_from else None
        self.memory_limit_gb = memory_limit_gb
        self.timeout_s = timeout_s
        self.command_runner = command_runner
        self.repo_root = Path(__file__).resolve().parents[2]
        self.base_config = self._load_config()
        model_engine = str(self.base_config.get("model_engine", "dp"))
        self.backend = "fsdp" if model_engine == "dp" else model_engine
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.resume_from is not None and not self.resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {self.resume_from}")

    def _load_config(self) -> dict[str, Any]:
        config_dir = Path(__file__).parents[1] / "trainer" / "config"
        # ``OmegaConf.load(ppo_trainer.yaml)`` is insufficient because VeRL's
        # component configs are selected through Hydra's defaults list.
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
            default_config = compose(config_name="ppo_trainer")
        OmegaConf.set_struct(default_config, False)
        user_config = OmegaConf.load(self.base_config_path)
        user_config.pop("defaults", None)
        merged = OmegaConf.merge(default_config, user_config)
        return OmegaConf.to_container(merged, resolve=True)

    def _trial_config(self, overrides: Mapping[str, Any], trial_dir: Path) -> dict[str, Any]:
        validate_overrides(overrides, backend=self.backend)
        config = deepcopy(self.base_config)
        for path, value in overrides.items():
            set_by_path(config, path, value)

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
            return ProbeResult(trial, dict(overrides), "timeout", duration_s=time.monotonic() - started, error=str(exc))

        duration = time.monotonic() - started
        (trial_dir / "stdout.log").write_text(completed.stdout or "")
        (trial_dir / "stderr.log").write_text(completed.stderr or "")
        if completed.returncode != 0:
            output = f"{completed.stdout}\n{completed.stderr}"
            status = "oom" if "out of memory" in output.lower() else "failed"
            return ProbeResult(trial, dict(overrides), status, duration_s=duration, error=output[-4000:])

        try:
            summary = summarize_probe(load_probe_states(trial_dir / "telemetry"), self.warmup_steps)
        except Exception as exc:
            return ProbeResult(trial, dict(overrides), "invalid_metrics", duration_s=duration, error=str(exc))
        status = "success" if summary["peak_memory_gb"] <= self.memory_limit_gb else "memory_limit"
        return ProbeResult(trial, dict(overrides), status, duration_s=duration, **summary)

    def search(self, search_space: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], list[ProbeResult]]:
        results = [self.run_probe(index, candidate) for index, candidate in enumerate(search_space)]
        feasible = [result for result in results if result.feasible]
        if not feasible:
            self.write_report(results, None)
            raise RuntimeError("No probe completed within the memory limit; see tuning_report.json")
        best = max(feasible, key=lambda result: (result.throughput, -result.step_time, result.gpu_utilization))
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
            param_offload = trial.suggest_categorical("param_offload", [False, True])
            overrides = {
                "actor_rollout_ref.actor.use_dynamic_bsz": True,
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": trial.suggest_categorical(
                    "token_budget", [2048, 4096, 8192, 12288]
                ),
                "actor_rollout_ref.model.enable_gradient_checkpointing": trial.suggest_categorical(
                    "gradient_checkpointing", [True, False]
                ),
                "actor_rollout_ref.actor.fsdp_config.param_offload": param_offload,
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload": trial.suggest_categorical(
                    "optimizer_offload", [False, True]
                )
                if param_offload
                else False,
            }
            result = self.run_probe(trial.number, overrides)
            results.append(result)
            trial.set_user_attr("probe_result", asdict(result))
            if not result.feasible:
                raise optuna.TrialPruned(result.status)
            return result.throughput

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        feasible = [result for result in results if result.feasible]
        if not feasible:
            self.write_report(results, None)
            raise RuntimeError("No Optuna probe completed within the memory limit")
        best = max(feasible, key=lambda result: (result.throughput, -result.step_time))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Athena restart-based offline tuning")
    parser.add_argument("--config", required=True, help="Base VeRL YAML config")
    parser.add_argument("--probe-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--work-dir", default="./athena_tuning")
    parser.add_argument("--resume-from", default=None, help="Optional global_step_N checkpoint shared by all trials")
    parser.add_argument(
        "--restart-requests",
        default=None,
        help="Optional restart_requests.jsonl emitted by AthenaTrainer; candidates are probed before defaults",
    )
    parser.add_argument("--memory-limit-gb", type=float, default=38.0)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--sampler", choices=("grid", "optuna"), default="grid")
    parser.add_argument("--trials", type=int, default=12, help="Number of trials for --sampler=optuna")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    tuner = OfflineAutoTuner(
        args.config,
        probe_steps=args.probe_steps,
        warmup_steps=args.warmup_steps,
        work_dir=args.work_dir,
        resume_from=args.resume_from,
        memory_limit_gb=args.memory_limit_gb,
        timeout_s=args.timeout_s,
    )
    if args.sampler == "optuna":
        best, _ = tuner.search_optuna(args.trials)
    else:
        search_space = load_restart_candidates(args.restart_requests) if args.restart_requests else []
        search_space.extend(build_default_fsdp_search_space(tuner.base_config))
        best, _ = tuner.search(search_space)
    output = tuner.apply_best_config(best, args.output)
    print(f"[Athena] Best configuration written to {output}")


if __name__ == "__main__":
    main()
