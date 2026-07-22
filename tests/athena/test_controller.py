from types import SimpleNamespace

import pytest

from verl.athena.controller import AthenaController
from verl.athena.parameters import ParameterLifecycle, get_parameter_spec, validate_overrides
from verl.athena.runtime_state import GPUStats, RuntimeState


def make_config(dynamic=True):
    return {
        "model_engine": "fsdp",
        "actor_rollout_ref": {
            "actor": {
                "use_dynamic_bsz": dynamic,
                "ppo_max_token_len_per_gpu": 4096,
                "ppo_micro_batch_size_per_gpu": 1,
            }
        },
        "trainer": {
            "athena": {
                "window_size": 2,
                "cooldown_steps": 1,
                "low_memory_threshold": 0.7,
                "low_gpu_utilization_threshold": 0.6,
            }
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
    controller = AthenaController(make_config(dynamic=False))
    controller.decide(make_state(1, 0.5, 0.4))
    actions = controller.decide(make_state(2, 0.5, 0.4))

    assert len(actions) == 1
    assert actions[0].requires_restart
    assert actions[0].candidate_config == {
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
    }


def test_backend_validation_rejects_megatron_parameter_on_fsdp():
    with pytest.raises(ValueError, match="not supported by backend fsdp"):
        validate_overrides(
            {"actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 2},
            backend="fsdp",
        )

    assert (
        get_parameter_spec("data.dataloader_num_workers").lifecycle
        == ParameterLifecycle.RESTART_REQUIRED
    )
