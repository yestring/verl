"""CPU-only tests for the entropy kernel configuration surface.

GPU numerics are exercised by ``verl/utils/kernel/benchmark.py``; this file
covers the tunable config plumbing (defaults, validation, name mappings) so
it runs on any machine without CUDA.
"""

import copy

import pytest

from verl.utils.kernel import kernels
from verl.utils.kernel.kernels import (
    BACKWARD_METHOD_NAMES,
    SUPPORTED_BACKWARD_METHODS,
    BackwardEnum,
    Config,
    configure,
    get_backward_method_by_name,
    get_backward_method_name,
    set_backward_method,
    set_forward_fused,
)


@pytest.fixture(autouse=True)
def _restore_module_config():
    saved = copy.deepcopy(kernels._config)
    yield
    kernels._config = saved


def test_config_defaults_reproduce_hardcoded_launch_values():
    config = Config()
    assert config.forward_vocab_per_split == 1024
    assert (config.forward_block_size_m, config.forward_block_size_n, config.forward_block_size_k) == (128, 256, 32)
    assert (config.forward_num_stages, config.forward_num_warps) == (3, 8)
    assert config.split_n_vocab_per_split == 9504
    assert (config.dlogits_block_size_m, config.dlogits_block_size_n, config.dlogits_block_size_k) == (128, 256, 32)
    assert config.fused_n_vocab_per_split == 16384
    assert config.grouped_n_group == 16
    assert (config.owner_compute_n_group, config.owner_compute_block_size_n) == (2, 128)
    assert (config.backward_block_size_m, config.backward_block_size_n, config.backward_block_size_k) == (128, 128, 32)
    assert (config.backward_num_stages, config.backward_num_warps) == (2, 8)
    assert (config.mn_block_size_m, config.mn_block_size_n, config.mn_block_size_k) == (128, 128, 32)
    assert (config.mn_num_stages, config.mn_num_warps) == (3, 8)
    assert (config.epilogue_block_size_m, config.epilogue_block_size_n) == (16, 64)
    assert config._backward == BackwardEnum._Split_Dlogits_N
    assert config._forward_fused is False


def test_backward_method_name_roundtrip():
    for name, value in BACKWARD_METHOD_NAMES.items():
        assert get_backward_method_by_name(name) == value
        assert get_backward_method_name(value) == name
    with pytest.raises(ValueError, match="Unknown backward method"):
        get_backward_method_by_name("not_a_method")
    with pytest.raises(ValueError, match="Unknown backward method value"):
        get_backward_method_name(-1)


def test_supported_backward_methods_excludes_unimplemented_and_unsized_strategies():
    # split_dlogits_m is never dispatched; total_fuse_mn exceeds A100 shared
    # memory for the training reduction (see the module-level comment).
    excluded = {"split_dlogits_m", "total_fuse_mn"}
    assert set(SUPPORTED_BACKWARD_METHODS) == set(BACKWARD_METHOD_NAMES) - excluded
    for name in excluded:
        assert name not in SUPPORTED_BACKWARD_METHODS
        assert get_backward_method_by_name(name) == BACKWARD_METHOD_NAMES[name]


def test_set_backward_method_accepts_name_and_value():
    set_backward_method("split_fused_n")
    assert kernels._config._backward == BackwardEnum._Split_Fused_N
    set_backward_method(BackwardEnum._Split_Dlogits_N)
    assert kernels._config._backward == BackwardEnum._Split_Dlogits_N
    with pytest.raises(ValueError):
        set_backward_method(99)
    with pytest.raises(ValueError):
        set_backward_method("bogus")


def test_set_forward_fused_type_check():
    set_forward_fused(True)
    assert kernels._config._forward_fused is True
    set_forward_fused(False)
    with pytest.raises(ValueError, match="bool"):
        set_forward_fused(1)


def test_configure_validates_and_applies_single_knobs():
    configure(vocab_per_split=2048, n_group=8, block_size_m=64, block_size_n=64, block_size_k=64)
    config = kernels._config
    assert config.forward_vocab_per_split == 2048
    assert config.split_n_vocab_per_split == 2048
    assert config.fused_n_vocab_per_split == 2048
    assert (config.grouped_n_group, config.owner_compute_n_group) == (8, 8)
    assert config.backward_block_size_m == 64
    assert config.forward_block_size_n == 64
    assert config.owner_compute_block_size_n == 64
    assert config.mn_block_size_k == 64
    with pytest.raises(ValueError, match="multiple of 128"):
        configure(vocab_per_split=1000)
    with pytest.raises(ValueError, match="power of two"):
        configure(block_size_m=96)
    with pytest.raises(ValueError, match="n_group"):
        configure(n_group=0)
