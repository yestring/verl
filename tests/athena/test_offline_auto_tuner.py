import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.athena.offline_auto_tuner import OfflineAutoTuner, ProbeResult, load_probe_states, summarize_probe


def state(step, throughput, step_time, peak_memory, utilization=0.5, mfu=0.0):
    return {
        "global_step": step,
        "throughput": throughput,
        "total_step_time": step_time,
        "gpu": {"memory_peak_gb": peak_memory, "utilization": utilization},
        "training": {"mfu": mfu},
    }


def test_structured_probe_metrics_discard_warmup_and_use_robust_aggregates(tmp_path: Path):
    path = tmp_path / "trace_step_1.jsonl"
    records = [
        state(1, 1, 99, 10),
        state(2, 100, 2.0, 20, 0.7, 0.4),
        state(3, 120, 1.5, 22, 0.9, 0.5),
        state(4, 110, 1.7, 21, 0.8, 0.45),
    ]
    path.write_text("\n".join(json.dumps(item) for item in records))

    summary = summarize_probe(load_probe_states(tmp_path), warmup_steps=1)
    assert summary["throughput"] == 110.0
    assert summary["step_time"] == 1.7
    assert summary["peak_memory_gb"] == 22.0
    assert summary["gpu_utilization"] == pytest.approx(0.8)
    assert summary["mfu"] == 0.45
    assert summary["samples"] == 3
    assert summary["metric_reliability"]["gpu.utilization"] == "partial"
    assert summary["metric_reliability"]["training.mfu"] == "legacy_unverified"


def test_missing_zero_metrics_are_nullable_and_marked_unreliable():
    records = [state(1, 10, 2, 3, utilization=0, mfu=0), state(2, 11, 1.9, 3, utilization=0, mfu=0)]
    summary = summarize_probe(records, warmup_steps=0)

    assert summary["mfu"] is None
    assert summary["gpu_utilization"] is None
    assert summary["ttft_ms"] is None
    assert summary["kv_usage"] is None
    assert summary["metric_reliability"]["training.mfu"] == "all_zero"
    assert summary["metric_reliability"]["rollout.ttft_ms"] == "all_zero"


class _FakeTrial:
    number = 0

    def __init__(self):
        self.suggestions = {}

    def suggest_categorical(self, name, choices):
        self.suggestions[name] = list(choices)
        return choices[0]

    def set_user_attr(self, _name, _value):
        pass


class _KernelTrial(_FakeTrial):
    """Fake trial that always enables fused kernels to exercise the kernel branch."""

    def suggest_categorical(self, name, choices):
        self.suggestions[name] = list(choices)
        return True if name == "use_fused_kernels" else choices[0]


def _run_fake_optuna(monkeypatch, backend, gpu_count, trial=None):
    if trial is None:
        trial = _FakeTrial()

    class Study:
        def optimize(self, objective, n_trials):
            assert n_trials == 1
            objective(trial)

    fake_optuna = SimpleNamespace(
        TrialPruned=type("TrialPruned", (Exception,), {}),
        samplers=SimpleNamespace(TPESampler=lambda seed: object()),
        create_study=lambda **_kwargs: Study(),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    tuner = OfflineAutoTuner.__new__(OfflineAutoTuner)
    tuner.backend = backend
    tuner.gpu_count = gpu_count
    tuner.utility_scorer = SimpleNamespace(apply_constraints=lambda _results: None)
    captured = []

    def run_probe(number, overrides):
        captured.append(overrides)
        return ProbeResult(number, dict(overrides), "success", throughput=1.0, gpu_count=gpu_count)

    tuner.run_probe = run_probe
    tuner._select_best = lambda results: results[0]
    tuner.write_report = lambda _results, _best: None
    tuner.search_optuna(1)
    return trial.suggestions, captured[0]


def test_optuna_fsdp_search_includes_sharding_and_workload_space(monkeypatch):
    suggestions, overrides = _run_fake_optuna(monkeypatch, "fsdp", 2)

    assert suggestions["sharding_strategy"] == ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]
    assert suggestions["train_batch_size"] == [4, 8, 16, 32]
    assert suggestions["rollout_n"] == [2, 4, 8]
    assert suggestions["rollout_max_num_seqs"] == [8, 16, 32]
    assert suggestions["use_fused_kernels"] == [False, True]
    assert overrides["actor_rollout_ref.actor.fsdp_config.sharding_strategy"] == "FULL_SHARD"
    assert overrides["actor_rollout_ref.actor.ppo_mini_batch_size"] <= overrides["data.train_batch_size"]
    # The default fake trial samples fused kernels off, so no kernel params leak.
    assert overrides["actor_rollout_ref.model.use_fused_kernels"] is False
    assert "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method" not in overrides


def test_optuna_megatron_search_respects_available_gpu_topology(monkeypatch):
    suggestions, overrides = _run_fake_optuna(monkeypatch, "megatron", 2)

    assert suggestions["tensor_model_parallel_size"] == [1, 2]
    assert suggestions["pipeline_model_parallel_size"] == [1, 2]
    tp = overrides["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"]
    pp = overrides["actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"]
    assert tp * pp <= 2


def test_optuna_kernel_sampling_targets_backend_engine_config(monkeypatch):
    from verl.utils.kernel.kernels import SUPPORTED_BACKWARD_METHODS

    suggestions, overrides = _run_fake_optuna(monkeypatch, "fsdp", 1, trial=_KernelTrial())

    assert suggestions["entropy_kernel_backward_method"] == list(SUPPORTED_BACKWARD_METHODS)
    assert suggestions["entropy_kernel_forward_fused"] == [False, True]
    assert suggestions["entropy_kernel_vocab_per_split"] == [4096, 8192, 16384, 32768]
    assert suggestions["entropy_kernel_n_group"] == [2, 4, 8, 16]
    assert overrides["actor_rollout_ref.model.use_fused_kernels"] is True
    assert (
        overrides["actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method"]
        in SUPPORTED_BACKWARD_METHODS
    )
    assert overrides["actor_rollout_ref.actor.fsdp_config.entropy_kernel_forward_fused"] is False
    assert overrides["actor_rollout_ref.actor.fsdp_config.entropy_kernel_vocab_per_split"] == 4096
    assert overrides["actor_rollout_ref.actor.fsdp_config.entropy_kernel_n_group"] == 2

    _, megatron_overrides = _run_fake_optuna(monkeypatch, "megatron", 1, trial=_KernelTrial())
    assert (
        megatron_overrides["actor_rollout_ref.actor.megatron.entropy_kernel_backward_method"]
        in SUPPORTED_BACKWARD_METHODS
    )
    assert megatron_overrides["actor_rollout_ref.actor.megatron.entropy_kernel_vocab_per_split"] == 4096
    assert "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method" not in megatron_overrides


def test_default_search_spaces_include_kernel_candidates():
    from verl.athena.offline_auto_tuner import build_default_fsdp_search_space, build_default_megatron_search_space

    fsdp_candidates = build_default_fsdp_search_space()
    kernel_candidates = [c for c in fsdp_candidates if c.get("actor_rollout_ref.model.use_fused_kernels")]
    assert len(kernel_candidates) == 6
    methods = {c["actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method"] for c in kernel_candidates}
    assert methods == {"split_dlogits_n", "split_fused_n", "split_fused_n_grouped", "owner_compute"}
    assert any(c.get("actor_rollout_ref.actor.fsdp_config.entropy_kernel_forward_fused") for c in kernel_candidates)
    assert any(c.get("actor_rollout_ref.actor.fsdp_config.entropy_kernel_n_group") == 8 for c in kernel_candidates)
    assert len({json.dumps(c, sort_keys=True) for c in fsdp_candidates}) == len(fsdp_candidates)

    megatron_candidates = build_default_megatron_search_space()
    kernel_candidates = [c for c in megatron_candidates if c.get("actor_rollout_ref.model.use_fused_kernels")]
    assert len(kernel_candidates) == 6
    assert all(c["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"] == 1 for c in kernel_candidates)
    assert all("actor_rollout_ref.actor.fsdp_config" not in c for c in kernel_candidates)


def test_trial_config_synchronizes_resolved_rollout_interpolation(tmp_path):
    tuner = OfflineAutoTuner.__new__(OfflineAutoTuner)
    tuner.backend = "megatron"
    tuner.resume_from = None
    tuner.warmup_steps = 1
    tuner.probe_steps = 2
    tuner.base_config = {
        "actor_rollout_ref": {
            "actor": {"rollout_n": 8},
            "ref": {"rollout_n": 8},
            "rollout": {"n": 8},
        },
        "trainer": {},
    }
    overrides = {"actor_rollout_ref.rollout.n": 2}

    config = tuner._trial_config(overrides, tmp_path)

    assert config["actor_rollout_ref"]["actor"]["rollout_n"] == 2
    assert config["actor_rollout_ref"]["ref"]["rollout_n"] == 2


def test_trial_config_writes_kernel_overrides_into_probe_config(tmp_path):
    tuner = OfflineAutoTuner.__new__(OfflineAutoTuner)
    tuner.backend = "fsdp"
    tuner.resume_from = None
    tuner.warmup_steps = 1
    tuner.probe_steps = 2
    tuner.base_config = {
        "actor_rollout_ref": {
            "actor": {"fsdp_config": {}},
            "model": {},
        },
        "trainer": {},
    }
    overrides = {
        "actor_rollout_ref.model.use_fused_kernels": True,
        "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method": "split_fused_n_grouped",
        "actor_rollout_ref.actor.fsdp_config.entropy_kernel_n_group": 8,
    }

    config = tuner._trial_config(overrides, tmp_path)

    assert config["actor_rollout_ref"]["model"]["use_fused_kernels"] is True
    assert config["actor_rollout_ref"]["actor"]["fsdp_config"]["entropy_kernel_backward_method"] == (
        "split_fused_n_grouped"
    )
    assert config["actor_rollout_ref"]["actor"]["fsdp_config"]["entropy_kernel_n_group"] == 8
