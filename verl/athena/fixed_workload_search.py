"""Compare seeded search methods with a fixed VeRL workload configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from itertools import product
from pathlib import Path

from verl.athena.offline_auto_tuner import OfflineAutoTuner, ProbeResult, _checkpoint_step, load_probe_states
from verl.athena.parameters import get_by_path, validate_overrides

TOKEN_BUDGET = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
CHECKPOINTING = "actor_rollout_ref.model.enable_gradient_checkpointing"
FUSED = "actor_rollout_ref.model.use_fused_kernels"


def measure_probe(tuner: OfflineAutoTuner, result: ProbeResult) -> tuple[tuple[int, ...], float]:
    states = load_probe_states(tuner.work_dir / f"trial_{result.trial:04d}" / "telemetry")
    first_step = _checkpoint_step(tuner.resume_from) + 1
    expected_steps = list(range(first_step, first_step + tuner.warmup_steps + tuner.probe_steps))
    steps = [int(state["global_step"]) for state in states]
    if steps != expected_steps:
        raise ValueError(f"Expected telemetry steps {expected_steps}, got {steps}")

    peaks = [float(state["gpu"]["memory_peak_gb"]) for state in states]
    measured_steps = states[tuner.warmup_steps :]
    tokens = tuple(int(state["total_num_tokens"]) for state in states)
    durations = [float(state["total_step_time"]) for state in measured_steps]
    if not all(math.isfinite(value) and value > 0 for value in peaks + durations) or min(tokens) <= 0:
        raise ValueError("Probe telemetry must report positive token counts, durations, and GPU peaks")

    result.peak_memory_gb = max(peaks)
    return tokens, sum(durations)


def check_rollout_cache(tuner: OfflineAutoTuner, required: bool) -> str:
    config = get_by_path(tuner.base_config, "skip.rollout_tq", {})
    if not config.get("enable", False):
        if required:
            raise ValueError("--require-cached-rollouts needs skip.rollout_tq.enable=true")
        return "live_rollout"
    if config.get("action") != "cache":
        raise ValueError("Fixed-workload studies require exact-step rollout cache, not repeat mode")

    from omegaconf import OmegaConf

    from verl.utils.skip.rollout_skip import RolloutTqSkip

    cache = RolloutTqSkip(OmegaConf.create(config), OmegaConf.create(tuner.base_config))
    first_step = _checkpoint_step(tuner.resume_from) + 1
    steps = range(first_step, first_step + tuner.warmup_steps + tuner.probe_steps)
    eligible = set(config["steps"])
    missing = [step for step in steps if step not in eligible or not cache.has_v1_cache(step)]
    if missing:
        raise ValueError(f"Exact-step rollout cache is incomplete for steps {missing}")
    return "exact_step_cached_rollout"


def check_staged_schedule(tuner: OfflineAutoTuner) -> None:
    config = tuner.base_config
    actor = get_by_path(config, "actor_rollout_ref.actor", {})
    roles = [("actor", tuner.backend, actor.get("optim") or {})]
    critic = get_by_path(config, "critic", {})
    critic_enabled = critic.get("enable")
    if critic_enabled is None:
        critic_enabled = get_by_path(config, "algorithm.adv_estimator", None) == "gae"
    if critic_enabled:
        roles.append(("critic", critic.get("strategy"), critic.get("optim") or {}))
    for name, strategy, optim in roles:
        if strategy == "megatron":
            decay_steps = optim.get("lr_decay_steps")
            stable_lr = optim.get("lr_decay_style", "constant") == "constant" or (
                decay_steps is not None and decay_steps > 0
            )
            fixed_decay = decay_steps is not None and decay_steps > 0
        else:
            scheduler = optim.get("warmup_style") or optim.get("lr_scheduler_type", "constant")
            stable_lr = scheduler == "constant"
            fixed_decay = False
        derived_warmup = (optim.get("lr_warmup_steps") or 0) <= 0 and optim.get("lr_warmup_steps_ratio", 0)
        if not stable_lr or (derived_warmup and not fixed_decay):
            raise ValueError(
                f"The {name} LR schedule changes with probe length. Use a constant schedule with "
                "fixed warm-up, or an explicit Megatron decay horizon."
            )


def run_search(args: argparse.Namespace) -> Path:
    staged = args.sampler == "staged"
    if not staged:
        import optuna

    args.work_dir.mkdir(parents=True, exist_ok=False)
    tuner = OfflineAutoTuner(
        args.config,
        probe_steps=args.probe_steps,
        warmup_steps=args.warmup_steps,
        work_dir=args.work_dir,
        resume_from=args.resume_from,
        memory_limit_gb=args.memory_limit_gib,
        timeout_s=args.timeout_s,
    )
    if not get_by_path(tuner.base_config, "trainer.use_v1", False) or (
        get_by_path(tuner.base_config, "trainer.v1.trainer_mode", None) != "athena_trainer"
    ):
        raise ValueError("Use a colocated config with trainer.v1.trainer_mode=athena_trainer")
    if tuner.backend not in {"fsdp", "megatron"} or tuner.gpu_count < 1:
        raise ValueError("The base configuration must allocate GPUs to FSDP or Megatron")
    if tuner.resume_from and get_by_path(tuner.base_config, "trainer.del_local_ckpt_after_load", False):
        raise ValueError("Keep trainer.del_local_ckpt_after_load=false so every probe shares the checkpoint")
    if any(
        get_by_path(tuner.base_config, path, False)
        for path in ("reward.reward_model.enable_resource_pool", "distillation.enable")
    ):
        raise ValueError("Use only the shared trainer GPU pool for GPU-second accounting")
    if not get_by_path(tuner.base_config, "actor_rollout_ref.actor.use_dynamic_bsz", False):
        raise ValueError("The base configuration must enable actor.use_dynamic_bsz")
    if args.sweep_checkpointing and tuner.backend != "fsdp":
        raise ValueError("--sweep-checkpointing uses the FSDP model flag; set Megatron recompute in the base config")
    kernel_path = (
        f"actor_rollout_ref.actor.{'megatron' if tuner.backend == 'megatron' else 'fsdp_config'}"
        ".entropy_kernel_backward_method"
    )
    if args.kernel_backward:
        if not get_by_path(tuner.base_config, FUSED, False):
            raise ValueError("--kernel-backward requires an already enabled fused path in the base configuration")
        from verl.utils.kernel.kernels import SUPPORTED_BACKWARD_METHODS

        unknown = set(args.kernel_backward) - set(SUPPORTED_BACKWARD_METHODS)
        if unknown:
            raise ValueError(f"Unsupported kernel choices: {sorted(unknown)}")
        validate_overrides({kernel_path: args.kernel_backward[0]}, backend=tuner.backend)
    rollout_mode = check_rollout_cache(tuner, args.require_cached_rollouts)

    budgets = sorted(set(args.token_budgets))
    kernel_choices = list(dict.fromkeys(args.kernel_backward))
    dimensions = [("token_budget", TOKEN_BUDGET, budgets)]
    if args.sweep_checkpointing:
        dimensions.append(("checkpointing", CHECKPOINTING, [False, True]))
    if kernel_choices:
        dimensions.append(("kernel_backward", kernel_path, kernel_choices))

    staged_plan = None
    if staged:
        check_staged_schedule(tuner)
        override_paths = [path for _, path, _ in dimensions]
        candidate_values = product(*(choices for _, _, choices in dimensions))
        candidates = [dict(zip(override_paths, values)) for values in candidate_values]
        screen_count = min(len(candidates), args.trials - args.promote_top_k)
        shortlist = random.Random(args.seed).sample(candidates, screen_count)
        staged_plan = {
            "unique_candidates": len(candidates),
            "shortlist_size": screen_count,
            "screen_steps": args.screen_steps,
            "promote_top_k": args.promote_top_k,
        }

    report = {
        "config": str(tuner.base_config_path),
        "config_sha256": hashlib.sha256(tuner.base_config_path.read_bytes()).hexdigest(),
        "resume_from": str(tuner.resume_from) if tuner.resume_from else None,
        "rollout_mode": rollout_mode,
        "sampler": args.sampler,
        "seed": args.seed,
        "requested_trials": args.trials,
        "warmup_steps": args.warmup_steps,
        "probe_steps": args.probe_steps,
        "timeout_s": args.timeout_s,
        "gpu_count": tuner.gpu_count,
        "objective": "processed_tokens_per_gpu_s",
        "constraints": tuner.utility_scorer.constraints,
        "staged_plan": staged_plan,
        "search_space": {
            "token_budgets": budgets,
            "sweep_checkpointing": args.sweep_checkpointing,
            "kernel_backward": kernel_choices,
        },
        "best_trial": None,
        "trials": [],
    }
    report_path = tuner.work_dir / "fixed_workload_search.json"
    best: ProbeResult | None = None
    best_score: float | None = None
    reference_token_counts: tuple[int, ...] = ()
    cumulative_gpu_s = 0.0

    def evaluate(
        trial_id: int, overrides: dict, phase: str, screen_trial: int | None = None
    ) -> tuple[float | None, ProbeResult]:
        nonlocal best, best_score, reference_token_counts, cumulative_gpu_s
        started = time.monotonic()
        try:
            result = tuner.run_probe(trial_id, overrides)
        except (OSError, ValueError) as exc:
            result = ProbeResult(trial_id, overrides, "failed", error=str(exc))
        gpu_s = tuner.gpu_count * (time.monotonic() - started)
        cumulative_gpu_s += gpu_s
        tokens = None
        token_counts = None
        step_s = None
        if result.status == "success":
            try:
                token_counts, step_s = measure_probe(tuner, result)
                if rollout_mode == "exact_step_cached_rollout":
                    overlap = min(len(reference_token_counts), len(token_counts))
                    if token_counts[:overlap] != reference_token_counts[:overlap]:
                        raise ValueError("Cached rollout's per-step token counts differ from earlier probes")
                    if len(token_counts) > len(reference_token_counts):
                        reference_token_counts = token_counts
                tokens = sum(token_counts[tuner.warmup_steps :])
            except (KeyError, OSError, TypeError, ValueError) as exc:
                result.status = "invalid_metrics"
                result.error = str(exc)
        tuner.utility_scorer.apply_constraints([result])
        score = tokens / (tuner.gpu_count * step_s) if result.feasible and tokens is not None else None
        if phase == "full" and score is not None and (best_score is None or score > best_score):
            best = result
            best_score = score
            report["best_trial"] = result.trial
        report["trials"].append(
            {
                "trial": result.trial,
                "overrides": overrides,
                "phase": phase,
                "fidelity_steps": tuner.probe_steps,
                "screen_trial": screen_trial,
                "status": result.status,
                "constraint_violations": result.constraint_violations,
                "error": result.error,
                "processed_tokens": tokens,
                "step_token_counts": token_counts,
                "measured_step_s": step_s,
                "processed_tokens_per_gpu_s": score,
                "step_time_p95_s": result.step_time_p95 if tokens is not None else None,
                "peak_allocated_gib": result.peak_memory_gb if tokens is not None else None,
                "probe_gpu_s": gpu_s,
                "cumulative_gpu_s": cumulative_gpu_s,
                "incumbent_processed_tokens_per_gpu_s": best_score,
            }
        )
        temporary_report = report_path.with_suffix(".json.tmp")
        temporary_report.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
        temporary_report.replace(report_path)
        return score, result

    if staged:
        ranked = []
        tuner.probe_steps = args.screen_steps
        try:
            for trial_id, overrides in enumerate(shortlist):
                score, _ = evaluate(trial_id, overrides, "screen")
                if score is not None:
                    ranked.append((score, trial_id, overrides))
        finally:
            tuner.probe_steps = args.probe_steps
        ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        for trial_id, (_, screen_trial, overrides) in enumerate(ranked[: args.promote_top_k], start=screen_count):
            evaluate(trial_id, overrides, "full", screen_trial)
    else:

        def objective(trial) -> float:
            overrides = {path: trial.suggest_categorical(name, choices) for name, path, choices in dimensions}
            score, result = evaluate(trial.number, overrides, "full")
            if score is None:
                raise optuna.TrialPruned(result.error or str(result.constraint_violations or result.status))
            return score

        sampler = (
            optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=max(1, args.trials // 3))
            if args.sampler == "tpe"
            else optuna.samplers.RandomSampler(seed=args.seed)
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=args.trials)
    if best is None:
        raise RuntimeError(f"No full probe met the constraints; see {report_path}")
    tuner.apply_best_config(best.config)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="A new directory for this study")
    parser.add_argument("--sampler", choices=("random", "tpe", "staged"), required=True)
    parser.add_argument("--token-budgets", type=int, nargs="+", required=True)
    parser.add_argument("--memory-limit-gib", type=float, required=True, help="Cap on the reported actor memory peak")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--require-cached-rollouts", action="store_true", help="Require exact-step V1 rollout replay")
    parser.add_argument("--seed", type=int, default=42, help="Search sampler seed")
    parser.add_argument("--trials", type=int, default=12, help="Maximum number of probe launches")
    parser.add_argument("--probe-steps", type=int, default=20)
    parser.add_argument("--screen-steps", type=int, help="Measured steps of each short probe for staged search")
    parser.add_argument("--promote-top-k", type=int, help="Maximum short-probe candidates to verify in staged search")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--sweep-checkpointing", action="store_true", help="Sweep the FSDP model checkpointing flag")
    parser.add_argument(
        "--kernel-backward", nargs="+", default=[], help="GPU-validated kernel choices on the fused path"
    )
    args = parser.parse_args()
    if min(args.trials, args.probe_steps, args.timeout_s, *args.token_budgets) <= 0 or args.warmup_steps < 0:
        parser.error("Trials, measured steps, timeouts, and token budgets must be positive; warm-up may be zero")
    if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0:
        parser.error("--memory-limit-gib must be finite and positive")
    if args.sampler == "staged":
        if args.screen_steps is None or not 0 < args.screen_steps < args.probe_steps:
            parser.error("staged search requires 0 < --screen-steps < --probe-steps")
        if args.promote_top_k is None or not 0 < args.promote_top_k < args.trials:
            parser.error("staged search requires 0 < --promote-top-k < --trials")
    else:
        if args.screen_steps is not None or args.promote_top_k is not None:
            parser.error("--screen-steps and --promote-top-k are available with --sampler staged")
        try:
            import optuna  # noqa: F401
        except ImportError:
            parser.error("Optuna is required for random and TPE. Install it in the existing VeRL GPU environment.")
    print(f"[Athena] Fixed-workload search report: {run_search(args)}")


if __name__ == "__main__":
    main()
