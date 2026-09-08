from types import SimpleNamespace

import pytest

from verl.athena.controller import AthenaController
from verl.athena.parameters import ParameterLifecycle, get_parameter_spec, validate_overrides
from verl.athena.runtime_state import GPUStats, RuntimeState, TrainingStats


def make_config(dynamic=True):
    return {
        "model_engine": "fsdp",
        "actor_rollout_ref": {
            "actor": {
                "strategy": "fsdp",
                "use_dynamic_bsz": dynamic,
                "ppo_max_token_len_per_gpu": 4096,
                "ppo_micro_batch_size_per_gpu": 1,
                "entropy_coeff": 0.02,
                "kl_loss_coef": 0.001,
                "use_kl_loss": True,
                "optim": {"lr": 1e-6},
            },
            "model": {"use_fused_kernels": False},
        },
        "trainer": {
            "save_freq": 20,
            "test_freq": 20,
            "total_training_steps": 10,
            "athena": {
                "window_size": 2,
                "cooldown_steps": 1,
                "low_memory_threshold": 0.7,
                "low_gpu_utilization_threshold": 0.6,
            },
        },
    }


def make_state(step, memory, utilization):
    return RuntimeState(global_step=step, gpu=GPUStats(memory_utilization=memory, utilization=utilization))


def test_dynamic_token_budget_is_applied_to_next_batch():
    config = make_config(dynamic=True)
    controller = AthenaController(config)
    assert controller.decide(make_state(1, 0.5, 0.4)) == []
    actions = controller.decide(make_state(2, 0.5, 0.4))

    assert len(actions) == 1
    assert actions[0].lifecycle == ParameterLifecycle.HOT
    assert actions[0].value == 5120

    batch = SimpleNamespace(extra_info={})
    controller.apply_hot_action(actions[0], batch)
    assert batch.extra_info["max_token_len_per_gpu"] == 5120
    assert config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == 5120


def test_fixed_batch_emits_restart_recommendation_instead_of_hot_change():
    config = make_config(dynamic=False)
    # Fused kernels already enabled: fall through to the micro-batch candidate.
    config["actor_rollout_ref"]["model"]["use_fused_kernels"] = True
    controller = AthenaController(config)
    controller.decide(make_state(1, 0.5, 0.4))
    actions = controller.decide(make_state(2, 0.5, 0.4))

    assert len(actions) == 1
    assert actions[0].requires_restart
    assert actions[0].candidate_config == {
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
    }


def test_cold_recommendation_emits_fused_kernel_candidate_when_disabled():
    controller = AthenaController(make_config(dynamic=False))
    controller.decide(make_state(1, 0.5, 0.4))
    actions = controller.decide(make_state(2, 0.5, 0.4))

    assert len(actions) == 1
    action = actions[0]
    assert action.requires_restart
    assert action.parameter == "actor_rollout_ref.model.use_fused_kernels"
    assert action.value is True
    assert action.candidate_config == {
        "actor_rollout_ref.model.use_fused_kernels": True,
        "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method": "split_fused_n_grouped",
        "actor_rollout_ref.actor.fsdp_config.entropy_kernel_forward_fused": True,
        "actor_rollout_ref.actor.fsdp_config.entropy_kernel_vocab_per_split": 16384,
        "actor_rollout_ref.actor.fsdp_config.entropy_kernel_n_group": 8,
    }


def test_cold_kernel_candidate_uses_megatron_engine_prefix():
    config = make_config(dynamic=False)
    config["actor_rollout_ref"]["actor"]["strategy"] = "megatron"
    controller = AthenaController(config)
    controller.decide(make_state(1, 0.5, 0.4))
    actions = controller.decide(make_state(2, 0.5, 0.4))

    candidate = actions[0].candidate_config
    assert "actor_rollout_ref.actor.megatron.entropy_kernel_backward_method" in candidate
    assert "actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method" not in candidate


KERNEL_FIELD_NAMES = (
    "entropy_kernel_backward_method",
    "entropy_kernel_forward_fused",
    "entropy_kernel_vocab_per_split",
    "entropy_kernel_n_group",
    "entropy_kernel_block_size_m",
    "entropy_kernel_block_size_n",
    "entropy_kernel_block_size_k",
)


def test_kernel_parameters_are_restart_required_and_backend_gated():
    for path in (
        "actor_rollout_ref.model.use_fused_kernels",
        *(f"actor_rollout_ref.actor.fsdp_config.{field}" for field in KERNEL_FIELD_NAMES),
        *(f"actor_rollout_ref.actor.megatron.{field}" for field in KERNEL_FIELD_NAMES),
    ):
        assert get_parameter_spec(path).lifecycle == ParameterLifecycle.RESTART_REQUIRED

    with pytest.raises(ValueError, match="not supported by backend megatron"):
        validate_overrides(
            {"actor_rollout_ref.actor.fsdp_config.entropy_kernel_backward_method": "split_fused_n"},
            backend="megatron",
        )
    with pytest.raises(ValueError, match="not supported by backend fsdp"):
        validate_overrides(
            {"actor_rollout_ref.actor.megatron.entropy_kernel_backward_method": "split_fused_n"},
            backend="fsdp",
        )
    # The master switch is backend-agnostic.
    validate_overrides({"actor_rollout_ref.model.use_fused_kernels": True}, backend="fsdp")
    validate_overrides({"actor_rollout_ref.model.use_fused_kernels": True}, backend="megatron")


def test_backend_validation_rejects_megatron_parameter_on_fsdp():
    with pytest.raises(ValueError, match="not supported by backend fsdp"):
        validate_overrides(
            {"actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 2},
            backend="fsdp",
        )

    assert get_parameter_spec("data.dataloader_num_workers").lifecycle == ParameterLifecycle.RESTART_REQUIRED


def test_algorithmic_hot_parameters_are_registered_and_injected_into_worker():
    calls = []
    trainer = SimpleNamespace(
        actor_rollout_wg=SimpleNamespace(apply_actor_runtime_overrides=lambda values: calls.append(values))
    )
    config = make_config()
    controller = AthenaController(config, trainer=trainer)
    state = make_state(8, 0.5, 0.8)
    state.training = TrainingStats(grad_norm=10.0, ppo_kl=0.03)
    controller.decide(make_state(7, 0.5, 0.8))
    actions = controller.decide(state)

    by_path = {action.parameter: action for action in actions}
    batch = SimpleNamespace(extra_info={})
    for path in (
        "actor_rollout_ref.actor.entropy_coeff",
        "actor_rollout_ref.actor.kl_loss_coef",
        "actor_rollout_ref.actor.optim.lr",
    ):
        assert get_parameter_spec(path).lifecycle == ParameterLifecycle.HOT
        controller.apply_hot_action(by_path[path], batch)

    assert calls == [{"entropy_coeff": 0.01}, {"kl_loss_coef": 0.0015}, {"lr": 8e-7}]
    assert config["actor_rollout_ref"]["actor"]["entropy_coeff"] == 0.01


def test_high_memory_increases_active_io_frequencies_with_lower_bound():
    config = make_config()
    config["trainer"]["save_freq"] = 2
    controller = AthenaController(config)
    controller.decide(make_state(1, 0.95, 0.8))
    actions = controller.decide(make_state(2, 0.95, 0.8))
    by_path = {action.parameter: action for action in actions}

    assert by_path["trainer.save_freq"].value == 10
    assert by_path["trainer.test_freq"].value == 40
