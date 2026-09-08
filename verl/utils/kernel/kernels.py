#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Implementations of the linear cross entropy with token entropy kernel.
"""

import typing
from dataclasses import dataclass

import torch
import torch.distributed as dist

from verl.utils.device import get_device_capability, get_device_name, is_cuda_available

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
    SUPPORT_CUDA_TMA = is_cuda_available and get_device_capability()[0] >= 9 and hasattr(tl, "make_tensor_descriptor")

except ImportError:
    HAVE_TRITON = False
    SUPPORT_CUDA_TMA = False

from verl.utils.device import get_torch_device

if not HAVE_TRITON:
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    @contextmanager
    def null_decorator(*args, **kwargs):
        if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
            return args[0]
        else:

            def inner(func):
                return func

            return inner

    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    tl = MagicMock()

elif SUPPORT_CUDA_TMA:
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: typing.Optional[int]):
        return torch.empty(size, device=get_device_name(), dtype=torch.int8)

    # https://github.com/triton-lang/triton/commit/43625fc968b693ab51884ca95adbcf3e43483fd0
    # Triton 3.5.0 stores allocators in ContextVar; values do not propagate to new
    # threads by default. Some execution paths in verl use thread pools (e.g.,
    # concurrent.futures), so we set a ContextVar *default* to avoid falling
    # back to NullAllocator in worker threads.
    try:
        import contextvars

        import triton.runtime._allocation as _triton_allocation

        if isinstance(getattr(_triton_allocation, "_allocator", None), contextvars.ContextVar):
            _triton_allocation._allocator = contextvars.ContextVar(
                _triton_allocation._allocator.name,
                default=alloc_fn,
            )
    except (ImportError, AttributeError):
        pass

    triton.set_allocator(alloc_fn)


@dataclass
class EntropyReductionEnum:
    """
    Enum for the reduction method of cross entropy.
    """

    _None = 0
    _Sum = 1
    _Mean = 2


def get_entropy_reduction_enum_number(reduction: str) -> int:
    """
    Get the enum number for the reduction method of cross entropy.
    """
    _enum = EntropyReductionEnum._None
    if reduction == "none":
        _enum = EntropyReductionEnum._None
    elif reduction == "sum":
        _enum = EntropyReductionEnum._Sum
    elif reduction == "mean":
        _enum = EntropyReductionEnum._Mean
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return _enum


def get_entropy_reduction_enum(ce_reduction: int) -> EntropyReductionEnum:
    """
    Get the enum for the reduction method of cross entropy.
    """
    _enum = EntropyReductionEnum._None
    if ce_reduction == 0:
        _enum = EntropyReductionEnum._None
    elif ce_reduction == 1:
        _enum = EntropyReductionEnum._Sum
    elif ce_reduction == 2:
        _enum = EntropyReductionEnum._Mean
    else:
        raise ValueError(f"Invalid ce_reduction: {ce_reduction}")
    return _enum


@dataclass
class BackwardEnum:
    """
    Enum for the backward method.
    """

    _Total_Fuse_MN = (
        0  # Fuse d_logits & d_hidden & d_weight, no intermediate storage, requires fp32 for d_hidden & d_weight
    )
    _Total_Separate = 1  # Store d_logits, no special requirements for d_hidden & d_weight
    _Split_Dlogits_N = 2  # split d_logits along its N dimension, aka. vocab_size
    _Split_Dlogits_M = 3  # split d_logits along its M dimension, aka. num_tokens
    _Split_Fused_N = 4
    _Split_Fused_N_Grouped = 5
    _Owner_Compute = 6


@dataclass
class Config:
    """Configuration for efficient entropy kernel operations.

    Every field defaults to the value that used to be hardcoded at the
    launch sites, so leaving the module config untouched reproduces the
    original behavior exactly. Block sizes are passed as constexpr kernel
    arguments and num_stages/num_warps as launch options, which makes them
    runtime tunable (each distinct signature is JIT-compiled once).
    """

    _backward: int = BackwardEnum._Split_Dlogits_N
    _use_triton: bool = True
    _forward_fused: bool = False
    # --- forward: mainloop + epilogue / fused forward ---
    forward_vocab_per_split: int = 1024
    forward_block_size_m: int = 128
    forward_block_size_n: int = 256
    forward_block_size_k: int = 32
    forward_num_stages: int = 3
    forward_num_warps: int = 8
    # --- backward: materialized d_logits (Total_Separate / Split_Dlogits_N) ---
    split_n_vocab_per_split: int = 9504
    dlogits_block_size_m: int = 128
    dlogits_block_size_n: int = 256
    dlogits_block_size_k: int = 32
    dlogits_group_size_m: int = 16
    dlogits_num_stages: int = 3
    dlogits_num_warps: int = 8
    # --- backward: fused split-N family (Split_Fused_N / Grouped / Owner_Compute) ---
    fused_n_vocab_per_split: int = 16384
    grouped_n_group: int = 16
    owner_compute_n_group: int = 2
    owner_compute_block_size_n: int = 128
    backward_block_size_m: int = 128
    backward_block_size_n: int = 128
    backward_block_size_k: int = 32
    backward_group_size_m: int = 16
    backward_num_stages: int = 2
    backward_num_warps: int = 8
    # --- backward: Total_Fuse_MN ---
    mn_block_size_m: int = 128
    mn_block_size_n: int = 128
    mn_block_size_k: int = 32
    mn_group_size_m: int = 16
    mn_num_stages: int = 3
    mn_num_warps: int = 8
    # --- epilogues ---
    epilogue_block_size_m: int = 16
    epilogue_block_size_n: int = 64


_config = Config()

BACKWARD_METHOD_NAMES: dict[str, int] = {
    "total_fuse_mn": BackwardEnum._Total_Fuse_MN,
    "total_separate": BackwardEnum._Total_Separate,
    "split_dlogits_n": BackwardEnum._Split_Dlogits_N,
    "split_dlogits_m": BackwardEnum._Split_Dlogits_M,
    "split_fused_n": BackwardEnum._Split_Fused_N,
    "split_fused_n_grouped": BackwardEnum._Split_Fused_N_Grouped,
    "owner_compute": BackwardEnum._Owner_Compute,
}

# Methods the backward dispatcher can actually run. ``split_dlogits_m`` is
# declared in BackwardEnum but its dispatch raises NotImplementedError, and
# ``total_fuse_mn`` exceeds A100 shared memory for the training reduction
# (its fp32 d_logits/d_weight tiles alone need ~128 KiB before pipelining),
# so it would crash the first real step; it stays selectable by explicit
# name on GPUs with larger shared memory.
SUPPORTED_BACKWARD_METHODS: tuple[str, ...] = (
    "total_separate",
    "split_dlogits_n",
    "split_fused_n",
    "split_fused_n_grouped",
    "owner_compute",
)


def get_backward_method_by_name(name: str) -> int:
    """Map a canonical strategy name to its BackwardEnum value."""
    try:
        return BACKWARD_METHOD_NAMES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown backward method {name!r}; supported={list(BACKWARD_METHOD_NAMES)}") from exc


def get_backward_method_name(value: int) -> str:
    """Map a BackwardEnum value to its canonical strategy name."""
    for name, method_value in BACKWARD_METHOD_NAMES.items():
        if method_value == value:
            return name
    raise ValueError(f"Unknown backward method value: {value}")


def _validate_power_of_two(name: str, value: int) -> None:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def configure(
    *,
    vocab_per_split: typing.Optional[int] = None,
    n_group: typing.Optional[int] = None,
    block_size_m: typing.Optional[int] = None,
    block_size_n: typing.Optional[int] = None,
    block_size_k: typing.Optional[int] = None,
) -> None:
    """Override kernel tunables on the module config.

    ``None`` leaves the current value untouched. ``vocab_per_split`` applies
    to the forward split, the materialized split-N strategy and the fused
    split-N family alike; ``n_group`` applies to the grouped and
    owner-compute strategies; the block sizes apply to every strategy
    family. Split sizes must be multiples of 128 and block sizes powers of
    two, matching the kernel assertions.
    """
    global _config
    if vocab_per_split is not None:
        if vocab_per_split % 128 != 0:
            raise ValueError(f"vocab_per_split must be a multiple of 128, got {vocab_per_split}")
        _config.forward_vocab_per_split = vocab_per_split
        _config.split_n_vocab_per_split = vocab_per_split
        _config.fused_n_vocab_per_split = vocab_per_split
    if n_group is not None:
        if n_group < 1:
            raise ValueError(f"n_group must be >= 1, got {n_group}")
        _config.grouped_n_group = n_group
        _config.owner_compute_n_group = n_group
    if block_size_m is not None:
        _validate_power_of_two("block_size_m", block_size_m)
        _config.backward_block_size_m = block_size_m
        _config.dlogits_block_size_m = block_size_m
        _config.mn_block_size_m = block_size_m
        _config.forward_block_size_m = block_size_m
    if block_size_n is not None:
        _validate_power_of_two("block_size_n", block_size_n)
        _config.backward_block_size_n = block_size_n
        _config.dlogits_block_size_n = block_size_n
        _config.mn_block_size_n = block_size_n
        _config.forward_block_size_n = block_size_n
        _config.owner_compute_block_size_n = block_size_n
    if block_size_k is not None:
        _validate_power_of_two("block_size_k", block_size_k)
        _config.backward_block_size_k = block_size_k
        _config.dlogits_block_size_k = block_size_k
        _config.mn_block_size_k = block_size_k
        _config.forward_block_size_k = block_size_k


def set_forward_fused(enable: bool):
    """Enable or disable fused forward kernel."""
    if not isinstance(enable, bool):
        raise ValueError(f"enable must be a bool, got {type(enable).__name__}")
    global _config
    _config._forward_fused = enable


def set_backward_method(backward_method: int | str):
    """
    Set the backward method.

    Accepts either a BackwardEnum value or one of the canonical strategy
    names in BACKWARD_METHOD_NAMES.
    """
    if isinstance(backward_method, str):
        backward_method = get_backward_method_by_name(backward_method)
    elif backward_method not in BACKWARD_METHOD_NAMES.values():
        raise ValueError(f"Unknown backward method value: {backward_method}")
    global _config
    _config._backward = backward_method


def warmup_entropy_kernels():
    """
    JIT-compile all entropy kernels at engine init instead of the first step.

    Every supported backward strategy (plus the fused forward path) is
    exercised once on tiny dummy tensors so Triton compilation happens here,
    outside of any profiled training step. The global ``_config`` is restored
    afterwards.
    """
    if not (HAVE_TRITON and torch.cuda.is_available()):
        return

    device = "cuda"
    num_tokens, hidden_size, vocab_size = 128, 1024, 2048
    hidden = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float32)
    weight = torch.randn((vocab_size, hidden_size), device=device, dtype=torch.float32)
    labels = torch.randint(0, vocab_size, (num_tokens,), device=device)

    saved = (_config._backward, _config._forward_fused, _config._use_triton)
    try:
        _config._use_triton = True
        # Compile the fused and mainloop forward once each (backward uses the
        # currently selected strategy below).
        for fused in (False, True):
            _config._forward_fused = fused
            logprobs, entropy, maximum, accu, entropy_b = efficient_entropy_forward(
                hidden, weight, labels, reduction=2
            )
            efficient_entropy_backward(
                torch.ones_like(logprobs),
                torch.ones_like(entropy),
                hidden,
                weight,
                labels,
                maximum,
                accu,
                entropy_b,
                reduction=2,
            )
        # 每个候选 backward 策略单独编译
        _config._forward_fused = False
        for method in SUPPORTED_BACKWARD_METHODS:
            _config._backward = method
            logprobs, entropy, maximum, accu, entropy_b = efficient_entropy_forward(
                hidden, weight, labels, reduction=2
            )
            efficient_entropy_backward(
                torch.ones_like(logprobs),
                torch.ones_like(entropy),
                hidden,
                weight,
                labels,
                maximum,
                accu,
                entropy_b,
                reduction=2,
            )
    finally:
        _config._backward, _config._forward_fused, _config._use_triton = saved


@triton.jit
def efficient_entropy_kernel_general_mainloop(
    rank,
    hidden_ptr,
    weight_ptr,
    labels_ptr,
    num_tokens,
    hidden_size,
    vocab_size,
    vocab_per_split,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    global_logprobs_scalar_ptr,
    rcp_temperature: tl.float32,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    forward mainloop
    """
    pid = tl.program_id(axis=0)
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if pid_m == 0 and pid_n == 0:
        tl.store(global_logprobs_scalar_ptr, 0.0)

    # create pointers for the first blocks of hidden
    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    else:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

    # load labels for this block
    labels = tl.load(labels_ptr + offs_am, mask=offs_am < num_tokens)

    # traverse over N dimension
    # _max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    _accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _logprobs = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    vocab_bound = min((pid_n + 1) * vocab_per_split, vocab_size)
    for n in range(0, num_pid_n):
        start_offs_bn = pid_n * vocab_per_split + n * BLOCK_SIZE_N
        offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if not USE_TMA:
            # weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)
            weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

        # iterate over K dimension
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            if USE_TMA:
                # load the next block of hidden and weight
                start_offs_k = k * BLOCK_SIZE_K
                _hidden = hidden_desc.load([start_offs_am, start_offs_k])
                _weight = weight_desc.load([start_offs_bn, start_offs_k])
            else:
                # load the next block of hidden and weight
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )

                _weight = tl.load(
                    weight_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                    & (offs_bn[:, None] < (min((pid_n + 1) * vocab_per_split, vocab_size))),
                    other=0.0,
                )

                # advance the ptrs to the next K block
                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            # GEMM
            logits = tl.dot(_hidden, _weight.trans(), logits)

        if not USE_TMA:
            # reset hidden_ptrs for next iteration
            hidden_ptrs -= hidden_size * stride_hidden_k

        # scale logits by temperature
        logits *= rcp_temperature

        logits_for_lse = tl.where(offs_bn[None, :] < vocab_bound, logits, float("-inf"))

        # update global maximum
        _max_old = _max
        m_pid_n = tl.max(logits_for_lse, axis=1)
        _max = tl.maximum(_max_old, m_pid_n)

        exp_logits = tl.exp(logits_for_lse - _max[:, None])
        coeff = tl.exp(_max_old - _max)
        _accu = coeff * _accu + tl.sum(exp_logits, axis=1)

        _entropy_b = _entropy_b * coeff + tl.sum(logits * exp_logits, axis=1)

        label_mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
        _logprobs += tl.sum(logits * label_mask, axis=1)

    # store maximum
    offs_max_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_max_n = pid_n
    maximum_ptrs = max_ptr + offs_max_n * stride_max_n + offs_max_m * stride_max_m
    tl.store(maximum_ptrs, _max, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits))

    # store entropy
    accu_ptrs = accu_ptr + offs_max_n * stride_accu_n + offs_max_m * stride_accu_m
    tl.store(accu_ptrs, _accu, mask=(offs_max_m < num_tokens) & (offs_max_n[None] < num_splits))
    entropy_b_ptrs = entropy_b_ptr + offs_max_n * stride_entropy_b_n + offs_max_m * stride_entropy_b_m
    tl.store(entropy_b_ptrs, _entropy_b, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits))
    # store logprobs
    vocab_left_idx = pid_n * vocab_per_split + rank * vocab_size
    vocab_right_idx = min((pid_n + 1) * vocab_per_split, vocab_size) + rank * vocab_size
    mask = (labels >= vocab_left_idx) & (labels < vocab_right_idx)
    mask &= offs_am < num_tokens
    global_logprobs_ptrs = global_logprobs_ptr + offs_am * stride_global_logprobs
    # tl.atomic_add(global_logprobs_ptrs, _logprobs, mask=mask)
    tl.store(global_logprobs_ptrs, _logprobs, mask=mask)








@triton.jit
def efficient_entropy_kernel_fused_forward(
    rank,
    hidden_ptr,
    weight_ptr,
    labels_ptr,
    num_tokens,
    hidden_size,
    vocab_size,
    vocab_per_split,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    Fused forward kernel: one CTA per (M-tile, vocab split).

    Computes the per-split partial statistics (maximum / accumulate /
    entropy_b) and the per-token label logit exactly like
    ``efficient_entropy_kernel_general_mainloop``, keeping the online
    softmax state in registers inside a single kernel. The final cross-split
    reduction is delegated to ``efficient_entropy_triton_kernel_epilogue``.
    """
    # ---------- 1. 当前 (M 块, split) 索引 ----------
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)

    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # ---------- 2. TMA 描述符或指针 ----------
    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    else:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

    # ---------- 3. 加载 labels ----------
    labels = tl.load(labels_ptr + offs_am, mask=offs_am < num_tokens)

    # ---------- 4. 初始化当前 split 的在线 LSE 状态 ----------
    _max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    _accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _logprobs = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    vocab_bound = min((pid_n + 1) * vocab_per_split, vocab_size)

    # ---------- 5. 遍历当前 split 内的所有 N 块 ----------
    for n in range(0, num_pid_n):
        start_offs_bn = pid_n * vocab_per_split + n * BLOCK_SIZE_N
        offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if not USE_TMA:
            weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

        # iterate over K dimension
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            if USE_TMA:
                # load the next block of hidden and weight
                start_offs_k = k * BLOCK_SIZE_K
                _hidden = hidden_desc.load([start_offs_am, start_offs_k])
                _weight = weight_desc.load([start_offs_bn, start_offs_k])
            else:
                # load the next block of hidden and weight
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )

                _weight = tl.load(
                    weight_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                    & (offs_bn[:, None] < vocab_bound),
                    other=0.0,
                )

                # advance the ptrs to the next K block
                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            # GEMM
            logits = tl.dot(_hidden, _weight.trans(), logits)

        if not USE_TMA:
            # reset hidden_ptrs for next iteration
            hidden_ptrs -= hidden_size * stride_hidden_k

        # scale logits by temperature
        logits *= rcp_temperature

        logits_for_lse = tl.where(offs_bn[None, :] < vocab_bound, logits, float("-inf"))

        # update global maximum
        _max_old = _max
        m_pid_n = tl.max(logits_for_lse, axis=1)
        _max = tl.maximum(_max_old, m_pid_n)

        exp_logits = tl.exp(logits_for_lse - _max[:, None])
        coeff = tl.exp(_max_old - _max)
        _accu = coeff * _accu + tl.sum(exp_logits, axis=1)

        _entropy_b = _entropy_b * coeff + tl.sum(logits * exp_logits, axis=1)

        label_mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
        _logprobs += tl.sum(logits * label_mask, axis=1)

    # ---------- 6. 存储该 (M 块, split) 的部分统计量 ----------
    offs_max_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_max_n = pid_n
    maximum_ptrs = max_ptr + offs_max_n * stride_max_n + offs_max_m * stride_max_m
    tl.store(maximum_ptrs, _max, mask=offs_max_m < num_tokens)

    accu_ptrs = accu_ptr + offs_max_n * stride_accu_n + offs_max_m * stride_accu_m
    tl.store(accu_ptrs, _accu, mask=offs_max_m < num_tokens)
    entropy_b_ptrs = entropy_b_ptr + offs_max_n * stride_entropy_b_n + offs_max_m * stride_entropy_b_m
    tl.store(entropy_b_ptrs, _entropy_b, mask=offs_max_m < num_tokens)

    # 每个 label 恰好属于一个 split，因此直接 store 无竞争
    vocab_left_idx = pid_n * vocab_per_split + rank * vocab_size
    vocab_right_idx = min((pid_n + 1) * vocab_per_split, vocab_size) + rank * vocab_size
    mask = (labels >= vocab_left_idx) & (labels < vocab_right_idx)
    mask &= offs_am < num_tokens
    global_logprobs_ptrs = global_logprobs_ptr + offs_am * stride_global_logprobs
    tl.store(global_logprobs_ptrs, _logprobs, mask=mask)




















    

@triton.jit
def efficient_entropy_triton_kernel_epilogue(
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    num_tokens,
    num_splits,
    global_max_ptr,
    stride_global_max: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    global_entropy_ptr,
    stride_global_entropy: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    global_logprobs_scalar_ptr,
    reduction: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    foward epilogue
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        max_ptrs = max_ptr + offs_m[:, None] * stride_max_m + offs_n[None, :] * stride_max_n

        _max = tl.load(max_ptrs, mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits), other=0.0)

        accu_ptrs = accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n
        _accu = tl.load(accu_ptrs, mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits), other=0.0)

        entropy_b_ptrs = entropy_b_ptr + offs_m[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n
        _entropy_b = tl.load(
            entropy_b_ptrs, mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits), other=0.0
        )

        # local reduction
        _max_old = global_max
        _local_max = tl.max(_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        _scale = tl.exp(_max - global_max[:, None])
        _coeff = tl.exp(_max_old - global_max)
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)
        global_entropy_b = _coeff * global_entropy_b + tl.sum(_scale * _entropy_b, axis=1)

    # store
    maximum_ptrs = global_max_ptr + offs_m * stride_global_max
    tl.store(maximum_ptrs, global_max, mask=offs_m < num_tokens)

    # store entropy_b
    global_entropy_b = tl.fdiv(global_entropy_b, global_accu)  # entropy_b
    tl.store(global_entropy_b_ptr + offs_m * stride_global_entropy_b, global_entropy_b, mask=offs_m < num_tokens)

    # store entropy
    global_accu_ptrs = global_accu_ptr + offs_m * stride_global_accu
    tl.store(global_accu_ptrs, global_accu, mask=offs_m < num_tokens)
    global_entropy = tl.log(global_accu) + global_max - global_entropy_b  # entropy_a
    global_entropy_ptrs = global_entropy_ptr + offs_m * stride_global_entropy
    tl.store(global_entropy_ptrs, global_entropy, mask=offs_m < num_tokens)
    # update logprobs
    global_logprobs_ptrs = global_logprobs_ptr + offs_m * stride_global_logprobs
    global_logprobs = tl.load(global_logprobs_ptrs, mask=offs_m < num_tokens)
    global_logprobs = global_max + tl.log(global_accu) - global_logprobs

    global_logprobs = -1 * global_logprobs
    if reduction == 0:
        tl.store(global_logprobs_ptrs, global_logprobs, mask=offs_m < num_tokens)
    elif reduction == 1:
        global_logprobs_scalar = tl.sum(global_logprobs, axis=0)
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)
    elif reduction == 2:
        global_logprobs_scalar = tl.sum(global_logprobs, axis=0) / num_tokens.to(tl.float32)
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)


@triton.jit
def efficient_entropy_triton_kernel_epilogue_tp(
    num_tokens,
    num_splits,
    reduced_max_ptr,
    stride_reduced_max_m: tl.int64,
    stride_reduced_max_n: tl.int64,
    original_max_ptr,
    stride_original_max_m: tl.int64,
    stride_original_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_max_ptr,
    stride_global_max: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        _reduced_max = tl.load(
            reduced_max_ptr + offs_m[:, None] * stride_reduced_max_m + offs_n[None, :] * stride_reduced_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        _original_max = tl.load(
            original_max_ptr + offs_m[:, None] * stride_original_max_m + offs_n[None, :] * stride_original_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        _accu = tl.load(
            accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )

        # local reduce-max
        _max_old = global_max
        _local_max = tl.max(_reduced_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        # update accumulate
        _coeff = tl.exp(_max_old - global_max)
        _scale = tl.exp(_original_max - global_max[:, None])
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

        # update entropy_b
        _entropy_b = tl.load(
            entropy_b_ptr + offs_m[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )
        global_entropy_b = _coeff * global_entropy_b + tl.sum(_scale * _entropy_b, axis=1)

    # store
    tl.store(global_max_ptr + offs_m * stride_global_max, global_max, mask=offs_m < num_tokens)
    tl.store(global_accu_ptr + offs_m * stride_global_accu, global_accu, mask=offs_m < num_tokens)
    tl.store(global_entropy_b_ptr + offs_m * stride_global_entropy_b, global_entropy_b, mask=offs_m < num_tokens)


@triton.jit
def efficient_entropy_triton_epilogue_tp_update(
    num_tokens,
    logprobs_ptr,
    stride_logprobs: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accumulate_ptr,
    stride_accumulate: tl.int64,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    entropy_ptr,
    stride_entropy: tl.int64,
    logprobs_scalar_ptr,
    reduction: int,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens)
    accumulate = tl.load(accumulate_ptr + offs_m * stride_accumulate, mask=offs_m < num_tokens)

    entropy_b = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens)
    entropy_b = tl.fdiv(entropy_b, accumulate)
    tl.store(entropy_b_ptr + offs_m * stride_entropy_b, entropy_b, mask=offs_m < num_tokens)

    entropy = tl.log(accumulate) + maximum - entropy_b
    tl.store(entropy_ptr + offs_m * stride_entropy, entropy, mask=offs_m < num_tokens)

    logprobs = tl.load(logprobs_ptr + offs_m * stride_logprobs, mask=offs_m < num_tokens)
    logprobs = maximum + tl.log(accumulate) - logprobs

    logprobs = -1 * logprobs
    if reduction == 0:
        tl.store(logprobs_ptr + offs_m * stride_logprobs, logprobs, mask=offs_m < num_tokens)
    elif reduction == 1:
        logprobs_scalar = tl.sum(logprobs, axis=0)
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)
    elif reduction == 2:
        logprobs_scalar = tl.sum(logprobs, axis=0) / num_tokens.to(tl.float32)
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)


_dedicated_stream, _dedicated_events = None, None


def efficient_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: typing.Optional[int] = 2,
    temperature: typing.Optional[float] = 1.0,
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
) -> list[torch.Tensor]:
    """
    forward host function
    """
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()

    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if dist_process_group is not None and not hasattr(efficient_entropy_forward, "_initialized"):
        global _dedicated_stream, _dedicated_events
        _dedicated_stream = get_torch_device().Stream(hidden.device)
        _dedicated_events = [get_torch_device().Event() for _ in range(2)]
        efficient_entropy_forward._initialized = True

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    vocab_size, hidden_size = weight.shape
    assert hidden_size % 128 == 0

    REDUCTION = get_entropy_reduction_enum(reduction)

    # ---------- 1. 分配最终输出张量（无论哪条路径都需要） ----------
    if REDUCTION == EntropyReductionEnum._None:
        if dist_process_group is None:
            logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        else:
            logprobs = torch.zeros((num_tokens,), device=hidden.device, dtype=torch.float32)
    elif REDUCTION in (EntropyReductionEnum._Sum, EntropyReductionEnum._Mean):
        # Zero-initialized: both the fused forward path and the epilogue accumulate
        # into this scalar with atomic adds and never store an initial value.
        logprobs = torch.zeros((), device=hidden.device, dtype=torch.float32)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    entropy = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    assert logprobs.is_contiguous() and entropy.is_contiguous()

    maximum = torch.empty_like(entropy)
    accumulate_and_entropy_b = torch.empty((num_tokens * 2,), device=hidden.device, dtype=torch.float32)
    accumulate_and_entropy_b_view = accumulate_and_entropy_b.view(2, num_tokens)
    accumulate = accumulate_and_entropy_b_view[0, :]
    entropy_b = accumulate_and_entropy_b_view[1, :]
    assert maximum.is_contiguous() and accumulate.is_contiguous() and entropy_b.is_contiguous()

    vocab_per_split = _config.forward_vocab_per_split
    assert vocab_per_split % 128 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    # ===================================================================
    # ========== 融合前向路径（Mainloop + Epilogue 融合） ==========
    # ===================================================================
    # 每个 (M-tile, split) 一个 CTA 写入部分统计量，跨 split 的最终归约复用
    # 下方共享的 epilogue。融合与主循环路径共用同一组部分统计量缓冲。
    _max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _accu = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _entropy_b = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)

    if REDUCTION == EntropyReductionEnum._None:
        _logprobs = logprobs
    else:
        _logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)

    assert _accu.is_contiguous() and _entropy_b.is_contiguous() and _max.is_contiguous()
    assert _accu.is_cuda and _entropy_b.is_cuda and _max.is_cuda

    if _config._use_triton:
        if _config._forward_fused and dist_process_group is None:
            # 融合前向：2D grid (M-tile, split)。每个 CTA 只计算一个 split 的
            # 部分统计量，跨 split 归约复用 epilogue。旧实现为 M-only grid，
            # 每个 CTA 串行遍历全部 split，并行度严重不足导致性能异常。
            grid = (triton.cdiv(num_tokens, _config.forward_block_size_m), num_splits)
            efficient_entropy_kernel_fused_forward[grid](
                _rank,
                hidden,
                weight,
                labels,
                num_tokens,
                hidden_size,
                vocab_size,
                vocab_per_split,
                hidden.stride(0),
                hidden.stride(1),
                weight.stride(0),
                weight.stride(1),
                _max,
                _max.stride(0),
                _max.stride(1),
                _accu,
                _accu.stride(0),
                _accu.stride(1),
                _entropy_b,
                _entropy_b.stride(0),
                _entropy_b.stride(1),
                _logprobs,
                _logprobs.stride(0),
                1.0 / temperature,
                BLOCK_SIZE_M=_config.forward_block_size_m,
                BLOCK_SIZE_N=_config.forward_block_size_n,
                BLOCK_SIZE_K=_config.forward_block_size_k,
                USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
                num_stages=_config.forward_num_stages,
                num_warps=_config.forward_num_warps,
            )
        else:
            # 1D kernel launch, then split the tile
            grid = (triton.cdiv(num_tokens, _config.forward_block_size_m) * num_splits,)
            efficient_entropy_kernel_general_mainloop[grid](
                _rank,
                hidden,
                weight,
                labels,
                num_tokens,
                hidden_size,
                vocab_size,
                vocab_per_split,
                hidden.stride(0),
                hidden.stride(1),
                weight.stride(0),
                weight.stride(1),
                _max,
                _max.stride(0),
                _max.stride(1),
                _accu,
                _accu.stride(0),
                _accu.stride(1),
                _entropy_b,
                _entropy_b.stride(0),
                _entropy_b.stride(1),
                _logprobs,
                _logprobs.stride(0),
                logprobs,
                1.0 / temperature,
                BLOCK_SIZE_M=_config.forward_block_size_m,
                BLOCK_SIZE_N=_config.forward_block_size_n,
                BLOCK_SIZE_K=_config.forward_block_size_k,
                USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
                num_stages=_config.forward_num_stages,
                num_warps=_config.forward_num_warps,
            )
    else:
        raise AssertionError("Triton is required for efficient entropy kernel")

    # reduction on maximum and maximum_indices
    epilogue_grid = (triton.cdiv(num_tokens, _config.epilogue_block_size_m),)

    if dist_process_group is None:
        efficient_entropy_triton_kernel_epilogue[epilogue_grid](
            _max,
            _max.stride(0),
            _max.stride(1),
            num_tokens,
            num_splits,
            maximum,
            maximum.stride(0),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            accumulate,
            accumulate.stride(0),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            entropy_b,
            entropy_b.stride(0),
            entropy,
            entropy.stride(0),
            _logprobs,
            _logprobs.stride(0),
            logprobs,
            REDUCTION,
            BLOCK_SIZE_M=_config.epilogue_block_size_m,
            BLOCK_SIZE_N=_config.epilogue_block_size_n,
            num_warps=4,
        )
    else:
        # tensor-parallel
        _max_backup = _max.clone()
        dist.all_reduce(_max, op=dist.ReduceOp.MAX, group=dist_process_group)

        get_torch_device().current_stream().record_event(_dedicated_events[0])
        with get_torch_device().stream(_dedicated_stream):
            _dedicated_stream.wait_event(_dedicated_events[0])
            dist.all_reduce(_logprobs, op=dist.ReduceOp.SUM, group=dist_process_group)
            _dedicated_stream.record_event(_dedicated_events[1])

        efficient_entropy_triton_kernel_epilogue_tp[epilogue_grid](
            num_tokens,
            num_splits,
            _max,
            _max.stride(0),
            _max.stride(1),
            _max_backup,
            _max_backup.stride(0),
            _max_backup.stride(1),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            maximum,
            maximum.stride(0),
            accumulate,
            accumulate.stride(0),
            entropy_b,
            entropy_b.stride(0),
            BLOCK_SIZE_M=_config.epilogue_block_size_m,
            BLOCK_SIZE_N=_config.epilogue_block_size_n,
            num_warps=4,
        )
        get_torch_device().current_stream().wait_event(_dedicated_events[1])

        dist.all_reduce(accumulate_and_entropy_b, op=dist.ReduceOp.SUM, group=dist_process_group)

        # update logprobs & entropy
        efficient_entropy_triton_epilogue_tp_update[epilogue_grid](
            num_tokens,
            _logprobs,
            _logprobs.stride(0),
            maximum,
            maximum.stride(0),
            accumulate,
            accumulate.stride(0),
            entropy_b,
            entropy_b.stride(0),
            entropy,
            entropy.stride(0),
            logprobs,
            REDUCTION,
            BLOCK_SIZE_M=_config.epilogue_block_size_m,
            num_warps=4,
        )

    return (logprobs, entropy, maximum, accumulate, entropy_b)


# NOTE: merge d_weight & d_hidden here, split along M & N
@triton.jit
def efficient_entropy_backward_kernel_general_mainloop_MN(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    backward mainloop, where d_logits & d_hidden & d_weight are fused
    """
    # block swizzling
    # pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    # pid_m = pid % num_pid_m
    # pid_n = pid // num_pid_m

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    start_offs_bn = pid_n * BLOCK_SIZE_N
    offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    maximum_ptrs = maximum_ptr + offs_am * stride_maximum
    maximum = tl.load(maximum_ptrs, mask=offs_am < num_tokens, other=0.0)
    accu_ptrs = accu_ptr + offs_am * stride_accu
    accu = tl.load(accu_ptrs, mask=offs_am < num_tokens, other=1e-6)  # epsilon to avoid division by zero
    accu_rcp = tl.fdiv(1.0, accu)

    d_entropy_ptrs = d_entropy_ptr + offs_am * stride_d_entropy
    d_entropy = tl.load(d_entropy_ptrs, mask=offs_am < num_tokens, other=0.0)
    if reduction == 0:  # none
        d_logprobs_ptrs = d_logprobs_ptr + offs_am * stride_d_logprobs
        d_logprobs = tl.load(d_logprobs_ptrs, mask=offs_am < num_tokens, other=0.0)
    elif reduction == 1:  # sum
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:  # mean
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs

    entropy_b_ptrs = entropy_b_ptr + offs_am * stride_entropy_b
    entropy_b = tl.load(entropy_b_ptrs, mask=offs_am < num_tokens, other=0.0)

    if not USE_TMA:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        # weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)
        weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
    labels_ptrs = labels_ptr + offs_am * stride_labels
    labels = tl.load(labels_ptrs, mask=offs_am < num_tokens, other=0)

    d_hidden_ptrs = d_hidden_ptr + offs_am[:, None] * stride_d_hidden_m + offs_k[None, :] * stride_d_hidden_k
    # d_weight_ptrs = d_weight_ptr + offs_k[:, None] * stride_d_weight_k + offs_bn[None, :] * stride_d_weight_n
    d_weight_ptrs = d_weight_ptr + offs_bn[:, None] * stride_d_weight_n + offs_k[None, :] * stride_d_weight_k

    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        if USE_TMA:
            start_offs_k = k * BLOCK_SIZE_K
            _hidden = hidden_desc.load([start_offs_am, start_offs_k])
            _weight = weight_desc.load([start_offs_bn, start_offs_k])
        else:
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_size),
                other=0.0,
            )
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k

        logits = tl.dot(_hidden, _weight.T, logits)

    if not USE_TMA:
        hidden_ptrs -= hidden_size * stride_hidden_k
        weight_ptrs -= hidden_size * stride_weight_k

    # scale logits by temperature
    logits *= rcp_temperature

    exp_logits = tl.exp(logits - maximum[:, None])

    mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
    d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
    d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])

    # scale d_logits by temperature
    d_logits *= rcp_temperature

    # loop for d_weight & d_hidden
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        start_offs_k = k * BLOCK_SIZE_K
        if USE_TMA:
            _hidden = hidden_desc.load([start_offs_am, start_offs_k])
        else:
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                other=0.0,
            )
            # _d_weight = tl.dot(tl.trans(_hidden).to(tl.float32), d_logits)
            # tl.atomic_add(d_weight_ptrs,
            #               _d_weight,
            #               mask=(offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[None, :] < vocab_size))
        _d_weight = tl.dot(d_logits.trans(), _hidden.to(tl.float32))
        tl.atomic_add(
            d_weight_ptrs,
            _d_weight,
            mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_size),
        )

        if USE_TMA:
            _weight = weight_desc.load([start_offs_bn, start_offs_k])
        else:
            # _weight = tl.load(
            #     weight_ptrs,
            #     mask=(offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[None, :] < vocab_size),
            #     other=0.0
            # )
            # _d_hidden = tl.dot(d_logits, tl.trans(_weight).to(tl.float32))
            _weight = tl.load(
                weight_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_size),
                other=0.0,
            )
        _d_hidden = tl.dot(d_logits, _weight.to(tl.float32))
        tl.atomic_add(
            d_hidden_ptrs,
            _d_hidden,
            mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
        )

        if not USE_TMA:
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        d_hidden_ptrs += BLOCK_SIZE_K * stride_d_hidden_k
        d_weight_ptrs += BLOCK_SIZE_K * stride_d_weight_k


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 16},
#             num_stages=3,
#             num_warps=8,
#         ),
#     ],
#     key=["num_tokens", "hidden_size", "vocab_size"],
# )
# @triton.jit
# def efficient_entropy_backward_kernel_d_hidden(
#     num_tokens: int,
#     hidden_size: int,
#     vocab_size: int,
#     rank: int,
#     hidden_ptr,
#     stride_hidden_m: tl.int64,
#     stride_hidden_k: tl.int64,
#     weight_ptr,
#     stride_weight_n: tl.int64,
#     stride_weight_k: tl.int64,
#     labels_ptr,
#     stride_labels: tl.int64,
#     maximum_ptr,
#     stride_maximum: tl.int64,
#     accu_ptr,
#     stride_accu: tl.int64,
#     d_entropy_ptr,
#     stride_d_entropy: tl.int64,
#     d_logprobs_ptr,
#     stride_d_logprobs: tl.int64,
#     reduction: int,
#     entropy_b_ptr,
#     stride_entropy_b: tl.int64,
#     d_hidden_ptr,
#     stride_d_hidden_m: tl.int64,
#     stride_d_hidden_k: tl.int64,
#     rcp_temperature: tl.float32,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     """
#     backward d_hidden
#     """
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
#     pid_m = pid % num_pid_m
#     pid_k = pid // num_pid_m

#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     result_offs_k = pid_k * BLOCK_SIZE_K + offs_k

#     maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens, other=0.0)
#     accu = tl.load(accu_ptr + offs_m * stride_accu, mask=offs_m < num_tokens, other=1e-6)
#     accu_rcp = tl.fdiv(1.0, accu)
#     d_entropy = tl.load(d_entropy_ptr + offs_m * stride_d_entropy, mask=offs_m < num_tokens, other=0.0)
#     if reduction == 0:
#         d_logprobs = tl.load(d_logprobs_ptr + offs_m * stride_d_logprobs, mask=offs_m < num_tokens, other=0.0)
#     elif reduction == 1:
#         d_logprobs = tl.load(d_logprobs_ptr)
#         d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
#     else:
#         d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
#         d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
#     d_logprobs = -1 * d_logprobs

#     entropy_b = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens, other=0.0)
#     labels = tl.load(labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=0)

#     # iterate over vocab_size
#     d_hidden = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
#     for n in range(0, tl.cdiv(vocab_size, BLOCK_SIZE_N)):
#         offs_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

#         hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
#         weight_ptrs = weight_ptr + (offs_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

#         # iterate over hidden_size to get logits
#         logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#         for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
#             _hidden = tl.load(
#                 hidden_ptrs,
#                 mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_m[:, None] < num_tokens),
#                 other=0.0,
#             )
#             _weight = tl.load(
#                 weight_ptrs,
#                 mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n[:, None] < vocab_size),
#                 other=0.0,
#             )

#             logits = tl.dot(_hidden, _weight.trans(), logits)

#             hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
#             weight_ptrs += BLOCK_SIZE_K * stride_weight_k

#         # scale logits by temperature
#         logits *= rcp_temperature

#         exp_logits = tl.exp(logits - maximum[:, None])

#         mask = (offs_n + rank * vocab_size)[None, :] == labels[:, None]
#         d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
#         d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])

#         # scale d_logits
#         d_logits *= rcp_temperature

#         # calculate d_hidden
#         weight_ptrs = weight_ptr + (offs_n[:, None] * stride_weight_n + result_offs_k[None, :] * stride_weight_k)
#         _weight = tl.load(
#             weight_ptrs, mask=(result_offs_k[None, :] < hidden_size) & (offs_n[:, None] < vocab_size), other=0.0
#         )
#         d_hidden = tl.dot(d_logits.to(weight_ptr.dtype.element_ty), _weight, d_hidden)

#     # write back
#     tl.store(
#         d_hidden_ptr + offs_m[:, None] * stride_d_hidden_m + result_offs_k[None, :] * stride_d_hidden_k,
#         d_hidden,
#         mask=(offs_m[:, None] < num_tokens) & (result_offs_k[None, :] < hidden_size),
#     )


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 16},
#             num_stages=3,
#             num_warps=8,
#         ),
#     ],
#     key=["num_tokens", "hidden_size", "vocab_size"],
# )
# @triton.jit
# def efficient_entropy_backward_kernel_d_weight(
#     num_tokens: int,
#     hidden_size: int,
#     vocab_size: int,
#     rank: int,
#     hidden_ptr,
#     stride_hidden_m: tl.int64,
#     stride_hidden_k: tl.int64,
#     weight_ptr,
#     stride_weight_n: tl.int64,
#     stride_weight_k: tl.int64,
#     labels_ptr,
#     stride_labels: tl.int64,
#     maximum_ptr,
#     stride_maximum: tl.int64,
#     accu_ptr,
#     stride_accu: tl.int64,
#     d_entropy_ptr,
#     stride_d_entropy: tl.int64,
#     d_logprobs_ptr,
#     stride_d_logprobs: tl.int64,
#     reduction: int,
#     entropy_b_ptr,
#     stride_entropy_b: tl.int64,
#     d_weight_ptr,
#     stride_d_weight_n: tl.int64,
#     stride_d_weight_k: tl.int64,
#     rcp_temperature: tl.float32,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
#     pid_n = pid % num_pid_n
#     pid_k = pid // num_pid_n

#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     result_offs_k = pid_k * BLOCK_SIZE_K + offs_k

#     d_weight = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
#     for m in range(0, tl.cdiv(num_tokens, BLOCK_SIZE_M)):
#         offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

#         maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens, other=0.0)
#         accu = tl.load(accu_ptr + offs_m * stride_accu, mask=offs_m < num_tokens, other=1e-6)
#         accu_rcp = tl.fdiv(1.0, accu)
#         d_entropy = tl.load(d_entropy_ptr + offs_m * stride_d_entropy, mask=offs_m < num_tokens, other=0.0)
#         if reduction == 0:
#             d_logprobs = tl.load(d_logprobs_ptr + offs_m * stride_d_logprobs, mask=offs_m < num_tokens, other=0.0)
#         elif reduction == 1:
#             d_logprobs = tl.load(d_logprobs_ptr)
#             d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
#         else:
#             d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
#             d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
#         d_logprobs = -1 * d_logprobs

#         entropy_b = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens, other=0.0)
#         labels = tl.load(labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=0)

#         hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
#         weight_ptrs = weight_ptr + (offs_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

#         logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#         for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
#             _hidden = tl.load(
#                 hidden_ptrs,
#                 mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_m[:, None] < num_tokens),
#                 other=0.0,
#             )
#             _weight = tl.load(
#                 weight_ptrs,
#                 mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n[:, None] < vocab_size),
#                 other=0.0,
#             )

#             logits = tl.dot(_hidden, _weight.trans(), logits)

#             hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
#             weight_ptrs += BLOCK_SIZE_K * stride_weight_k

#         logits *= rcp_temperature

#         exp_logits = tl.exp(logits - maximum[:, None])

#         mask = (offs_n + rank * vocab_size)[None, :] == labels[:, None]
#         d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
#         d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])

#         d_logits *= rcp_temperature

#         hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + result_offs_k[None, :] * stride_hidden_k)
#         _hidden = tl.load(
#             hidden_ptrs, mask=(result_offs_k[None, :] < hidden_size) & (offs_m[:, None] < num_tokens), other=0.0
#         )
#         d_weight = tl.dot(d_logits.to(d_weight_ptr.dtype.element_ty).trans(), _hidden, d_weight)

#     # write back
#     tl.store(
#         d_weight_ptr + offs_n[:, None] * stride_d_weight_n + result_offs_k[None, :] * stride_d_weight_k,
#         d_weight,
#         mask=(offs_n[:, None] < vocab_size) & (result_offs_k[None, :] < hidden_size),
#     )


# NOTE: split tile from d_logits' perspective
@triton.jit
def efficient_entropy_backward_kernel_general_d_logits(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    backward d_logits
    """
    # block swizzling
    # pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    # pid_m = pid % num_pid_m
    # pid_n = pid // num_pid_m

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    start_offs_bn = pid_n * BLOCK_SIZE_N
    offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum_ptrs = maximum_ptr + offs_am * stride_maximum
    maximum = tl.load(maximum_ptrs, mask=offs_am < num_tokens, other=0.0)
    accu_ptrs = accu_ptr + offs_am * stride_accu
    accu = tl.load(accu_ptrs, mask=offs_am < num_tokens, other=1e-6)  # epsilon to avoid division by zero
    accu_rcp = tl.fdiv(1.0, accu)

    d_entropy_ptrs = d_entropy_ptr + offs_am * stride_d_entropy
    d_entropy = tl.load(d_entropy_ptrs, mask=offs_am < num_tokens, other=0.0)
    if reduction == 0:  # none
        d_logprobs_ptrs = d_logprobs_ptr + offs_am * stride_d_logprobs
        d_logprobs = tl.load(d_logprobs_ptrs, mask=offs_am < num_tokens, other=0.0)
    elif reduction == 1:  # sum
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:  # mean
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs

    entropy_b_ptrs = entropy_b_ptr + offs_am * stride_entropy_b
    entropy_b = tl.load(entropy_b_ptrs, mask=offs_am < num_tokens, other=0.0)

    labels_ptrs = labels_ptr + offs_am * stride_labels
    labels = tl.load(labels_ptrs, mask=offs_am < num_tokens, other=0)

    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    else:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        # weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)
        weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        if USE_TMA:
            start_offs_k = k * BLOCK_SIZE_K
            _hidden = hidden_desc.load([start_offs_am, start_offs_k])
            _weight = weight_desc.load([start_offs_bn, start_offs_k])
        else:
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_size),
                other=0.0,
            )
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        logits = tl.dot(_hidden, _weight.T, logits)

    if not USE_TMA:
        hidden_ptrs -= hidden_size * stride_hidden_k
        weight_ptrs -= hidden_size * stride_weight_k

    # scale logits by temperature
    logits *= rcp_temperature

    exp_logits = tl.exp(logits - maximum[:, None])

    mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
    d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
    d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])

    # scale d_logits by temperature
    d_logits *= rcp_temperature

    # store d_logits
    d_logits_ptrs = d_logits_ptr + offs_am[:, None] * stride_d_logits_m + offs_bn[None, :] * stride_d_logits_n
    tl.store(
        d_logits_ptrs,
        d_logits,  # will be implicitly converted to d_logits_ptrs.dtype.element_ty
        mask=(offs_am[:, None] < num_tokens) & (offs_bn[None, :] < vocab_size),
    )


@triton.jit
def efficient_entropy_backward_kernel_general_d_logits_split_N(
    split_idx: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    vocab_per_split: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    start_offs_bn = split_idx * vocab_per_split + pid_n * BLOCK_SIZE_N
    offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum = tl.load(maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0)
    accu = tl.load(accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6)
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0)
    if reduction == 0:
        d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs, mask=offs_am < num_tokens, other=0.0)
    elif reduction == 1:
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs
    entropy_b = tl.load(entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0)
    labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)

    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    else:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
        vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)

    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        if USE_TMA:
            start_offs_k = k * BLOCK_SIZE_K
            _hidden = hidden_desc.load([start_offs_am, start_offs_k])
            _weight = weight_desc.load([start_offs_bn, start_offs_k])
        else:
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_right_bound),
                other=0.0,
            )
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        logits = tl.dot(_hidden, _weight.T, logits)

    logits *= rcp_temperature
    exp_logits = tl.exp(logits - maximum[:, None])

    mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
    d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
    d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])

    d_logits *= rcp_temperature

    # filter d_logits with mask
    result_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_am[:, None] < num_tokens) & (result_offs_n[None, :] < vocab_per_split)

    tl.store(
        d_logits_ptr + offs_am[:, None] * stride_d_logits_m + result_offs_n[None, :] * stride_d_logits_n, d_logits, mask
    )




#############################################################################################################fused

@triton.jit
def efficient_entropy_backward_kernel_fused_split_n(
    split_idx: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    vocab_per_split: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    融合的 Split-N 反向 Kernel（寄存器优化版）：
    1. 计算当前 N-split 的 logits（寄存器中）
    2. 计算 d_logits（寄存器中）
    3. 使用累加器在寄存器中累积所有 K 块的 d_hidden 和 d_weight
    4. 循环结束后，一次性原子加回全局内存

    """
    # ---------- 1. 获取当前 Program 的 ID 和偏移 ----------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    start_offs_bn = split_idx * vocab_per_split + pid_n * BLOCK_SIZE_N
    offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # ---------- 2. 加载前向统计量和梯度 ----------
    maximum = tl.load(maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0)
    accu = tl.load(accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6)
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0)

    if reduction == 0:
        d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs, mask=offs_am < num_tokens, other=0.0)
    elif reduction == 1:
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:  # reduction == 2 (mean)
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs

    entropy_b = tl.load(entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0)
    labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)

    vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)

    # ---------- 3. 计算 logits（分块 GEMM，寄存器累加） ----------
    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_TMA:
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    else:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        if USE_TMA:
            start_offs_k = k * BLOCK_SIZE_K
            _hidden = hidden_desc.load([start_offs_am, start_offs_k])
            _weight = weight_desc.load([start_offs_bn, start_offs_k])
        else:
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_right_bound),
                other=0.0,
            )
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        logits = tl.dot(_hidden, _weight.T, logits)

    # ---------- 4. 温度缩放 + 数值稳定 ----------
    logits *= rcp_temperature
    exp_logits = tl.exp(logits - maximum[:, None])

    # ---------- 5. 计算 d_logits（寄存器中，不写回） ----------
    mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
    d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
    d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])
    d_logits *= rcp_temperature

    # ---------- 6. 梯度计算（使用寄存器累加器！） ----------
    # 初始化累加器（寄存器中累积所有 K 块的贡献）
    d_hidden_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    d_weight_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # 重置指针用于梯度计算
    if not USE_TMA:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        if USE_TMA:
            start_offs_k = k * BLOCK_SIZE_K
            _hidden = hidden_desc.load([start_offs_am, start_offs_k])
            _weight = weight_desc.load([start_offs_bn, start_offs_k])
        else:
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_right_bound),
                other=0.0,
            )
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k

        # 在寄存器中累加梯度
        d_hidden_acc = tl.dot(d_logits, _weight.to(tl.float32), d_hidden_acc)
        d_weight_acc = tl.dot(d_logits.trans(), _hidden.to(tl.float32), d_weight_acc)

    # ---------- 7. 一次性写回全局内存（原子加） ----------
    # 只有一次原子操作，大幅减少冲突！
    d_hidden_ptrs = d_hidden_ptr + offs_am[:, None] * stride_d_hidden_m + offs_k[None, :] * stride_d_hidden_k
    d_weight_ptrs = d_weight_ptr + offs_bn[:, None] * stride_d_weight_n + offs_k[None, :] * stride_d_weight_k

    tl.atomic_add(
        d_hidden_ptrs,
        d_hidden_acc,
        mask=(offs_am[:, None] < num_tokens) & (offs_k[None, :] < hidden_size)
    )
    tl.atomic_add(
        d_weight_ptrs,
        d_weight_acc,
        mask=(offs_bn[:, None] < vocab_right_bound) & (offs_k[None, :] < hidden_size)
    )






@triton.jit
def efficient_entropy_backward_kernel_fused_split_n_grouped(
    group_idx: int,
    N_GROUP: tl.constexpr,   # 编译时常量，例如 2, 4, 8
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    vocab_per_split: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    Fused Split-N Backward with N-Group Fusion.
    Each CTA processes N_GROUP consecutive vocab splits, reusing the same hidden tile.
    """
    # ---------- 1. 获取当前 Program 的 ID 和偏移量 ----------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # ---------- 2. 加载前向统计量和损失梯度（所有 N_GROUP 共享） ----------
    maximum = tl.load(maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0)
    accu = tl.load(accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6)
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0)
    if reduction == 0:
        d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs, mask=offs_am < num_tokens, other=0.0)
    elif reduction == 1:
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:  # reduction == 2 (mean)
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs
    entropy_b = tl.load(entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0)
    labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)

    # ---------- 3. 构建 Hidden 描述符或指针（只构建一次） ----------
    if USE_TMA:
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
    else:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

    # ---------- 4. 初始化 d_hidden 累加器（寄存器内累积所有 N_GROUP） ----------
    d_hidden_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # ---------- 5. 循环处理 N_GROUP 个 split ----------
    for ng in range(N_GROUP):
        split_idx = group_idx * N_GROUP + ng
        base_vocab = split_idx * vocab_per_split
        # 只处理有效的 split（不超出 vocab_size）
        if base_vocab < vocab_size:
            vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)

            # ---------- 5a. 计算当前 split 的 logits ----------
            logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            if USE_TMA:
                weight_desc = tl.make_tensor_descriptor(
                    weight_ptr,
                    shape=[vocab_size, hidden_size],
                    strides=[stride_weight_n, 1],
                    block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
                )
                for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                    start_offs_k = k * BLOCK_SIZE_K
                    _hidden = hidden_desc.load([pid_m * BLOCK_SIZE_M, start_offs_k])
                    _weight = weight_desc.load([base_vocab + pid_n * BLOCK_SIZE_N, start_offs_k])
                    logits = tl.dot(_hidden, _weight.T, logits)
            else:
                weight_ptrs = weight_ptr + (
                    (base_vocab + offs_bn)[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
                )
                # 保存 hidden_ptrs 的原始位置以便每次循环重置
                hidden_ptrs_orig = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
                for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                    _hidden = tl.load(
                        hidden_ptrs_orig,
                        mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                        other=0.0,
                    )
                    _weight = tl.load(
                        weight_ptrs,
                        mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                        & (offs_bn[:, None] < vocab_right_bound - base_vocab),
                        other=0.0,
                    )
                    logits = tl.dot(_hidden, _weight.T, logits)
                    hidden_ptrs_orig += BLOCK_SIZE_K * stride_hidden_k
                    weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            logits *= rcp_temperature
            exp_logits = tl.exp(logits - maximum[:, None])

            # ---------- 5b. 计算 d_logits（寄存器中） ----------
            # labels 是全局词表索引，TP 下需要加上本 rank 的 vocab 偏移
            mask = (base_vocab + offs_bn + rank * vocab_size)[None, :] == labels[:, None]
            d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
            d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])
            d_logits *= rcp_temperature

            # ---------- 5c. 累加 d_hidden 和计算 d_weight ----------
            weight_ptrs = weight_ptr + (
                (base_vocab + offs_bn)[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
            )
            hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
            d_weight_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

            for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
                _weight = tl.load(
                    weight_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                    & (offs_bn[:, None] < vocab_right_bound - base_vocab),
                    other=0.0,
                )
                d_hidden_acc = tl.dot(d_logits, _weight.to(tl.float32), d_hidden_acc)
                d_weight_acc = tl.dot(d_logits.trans(), _hidden.to(tl.float32), d_weight_acc)
                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            # ---------- 5d. 写回 d_weight（直接 store，因为当前 N 区间唯一） ----------
            d_weight_ptrs = d_weight_ptr + (
                (base_vocab + offs_bn)[:, None] * stride_d_weight_n + offs_k[None, :] * stride_d_weight_k
            )
            tl.store(
                d_weight_ptrs,
                d_weight_acc,
                mask=(offs_bn[:, None] < vocab_right_bound - base_vocab) & (offs_k[None, :] < hidden_size)
            )

    # ---------- 6. 所有 N_GROUP 处理完毕后，一次性原子加 d_hidden ----------
    d_hidden_ptrs = d_hidden_ptr + offs_am[:, None] * stride_d_hidden_m + offs_k[None, :] * stride_d_hidden_k
    tl.atomic_add(
        d_hidden_ptrs,
        d_hidden_acc,
        mask=(offs_am[:, None] < num_tokens) & (offs_k[None, :] < hidden_size)
    )




@triton.jit
def efficient_entropy_backward_kernel_owner_compute(
    group_start_idx: int,          # 当前 CTA 处理的 N-split 起始索引
    N_GROUP: tl.constexpr,         # 每个 CTA 处理的 N-split 数量（如 2, 4, 8）
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    Partial Owner-Compute Backward:
    - grid = M_tiles × N_groups (保持高度并行)
    - 每个 CTA 处理 N_GROUP 个连续的 N-split
    - 在寄存器中累积这 N_GROUP 个 split 的 d_hidden
    - 最后 atomic_add 一次 d_hidden
    - d_weight 直接 store
    """
    # ---------- 1. 获取当前 CTA 的 M 和 N 分组 ID ----------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n_group = tl.cdiv(tl.cdiv(vocab_size, BLOCK_SIZE_N), N_GROUP)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n_group
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n_group = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # ---------- 2. 加载前向统计量 ----------
    maximum = tl.load(maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0)
    accu = tl.load(accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6)
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0)
    if reduction == 0:
        d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs, mask=offs_am < num_tokens, other=0.0)
    elif reduction == 1:
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs
    entropy_b = tl.load(entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0)
    labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)

    # ---------- 3. 初始化 d_hidden 累加器 ----------
    d_hidden_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # ---------- 4. 处理 N_GROUP 个连续的 N-split ----------
    hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

    for ng in range(N_GROUP):
        n_idx = pid_n_group * N_GROUP + ng
        offs_bn = n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        vocab_right_bound = min((n_idx + 1) * BLOCK_SIZE_N, vocab_size)

        # 只处理有效的 split
        if n_idx * BLOCK_SIZE_N < vocab_size:
            # ---------- 4a. 计算 logits ----------
            logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
            hidden_ptrs_orig = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

            for k in range(tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                _hidden = tl.load(
                    hidden_ptrs_orig,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
                _weight = tl.load(
                    weight_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_right_bound),
                    other=0.0,
                )
                logits = tl.dot(_hidden, _weight.T, logits)
                hidden_ptrs_orig += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            logits *= rcp_temperature
            exp_logits = tl.exp(logits - maximum[:, None])

            mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
            d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
            d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])
            d_logits *= rcp_temperature

            # ---------- 4b. 累加 d_hidden + 计算 d_weight ----------
            weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
            hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
            d_weight_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

            for k in range(tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
                _weight = tl.load(
                    weight_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_right_bound),
                    other=0.0,
                )
                d_hidden_acc = tl.dot(d_logits, _weight.to(tl.float32), d_hidden_acc)
                d_weight_acc = tl.dot(d_logits.trans(), _hidden.to(tl.float32), d_weight_acc)
                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            # ---------- 4c. 写回 d_weight（直接 store） ----------
            d_weight_ptrs = d_weight_ptr + offs_bn[:, None] * stride_d_weight_n + offs_k[None, :] * stride_d_weight_k
            tl.store(
                d_weight_ptrs,
                d_weight_acc,
                mask=(offs_bn[:, None] < vocab_right_bound) & (offs_k[None, :] < hidden_size)
            )

    # ---------- 5. 一次性原子加 d_hidden ----------
    d_hidden_ptrs = d_hidden_ptr + offs_am[:, None] * stride_d_hidden_m + offs_k[None, :] * stride_d_hidden_k
    tl.atomic_add(
        d_hidden_ptrs,
        d_hidden_acc,
        mask=(offs_am[:, None] < num_tokens) & (offs_k[None, :] < hidden_size)
    )





























def efficient_entropy_backward(
    dlogprobs: torch.Tensor,
    dentropy: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    acc: torch.Tensor,
    entropy_b: torch.Tensor,
    reduction: typing.Optional[int] = 2,
    should_return_fp32_grad: bool = False,
    temperature: typing.Optional[float] = 1.0,
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
) -> list[torch.Tensor]:
    """
    backward host function
    """
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    vocab_size, hidden_size = weight.shape
    assert hidden_size % 128 == 0

    REDUCTION = get_entropy_reduction_enum(reduction)

    if REDUCTION == EntropyReductionEnum._None:
        assert dlogprobs.shape == (num_tokens,)
    else:
        assert dlogprobs.dim() == 0

    assert dlogprobs.is_contiguous() and dentropy.is_contiguous()
    assert dlogprobs.is_cuda and dentropy.is_cuda
    assert dlogprobs.device == hidden.device and dlogprobs.device == dentropy.device
    assert dentropy.shape == (num_tokens,)

    d_hidden, d_weight = None, None
    if (
        _config._backward == BackwardEnum._Total_Fuse_MN
        or _config._backward == BackwardEnum._Split_Fused_N
        or _config._backward == BackwardEnum._Split_Fused_N_Grouped
        or _config._backward == BackwardEnum._Owner_Compute
        or should_return_fp32_grad
    ):
        d_hidden = torch.zeros_like(hidden, dtype=torch.float32, device=hidden.device)
        d_weight = torch.zeros_like(weight, dtype=torch.float32, device=weight.device)
    else:
        d_hidden = torch.empty_like(hidden, dtype=hidden.dtype, device=hidden.device)
        d_weight = torch.empty_like(weight, dtype=hidden.dtype, device=weight.device)
    assert d_hidden.is_contiguous() and d_weight.is_contiguous()

    assert maximum.is_contiguous() and acc.is_contiguous()
    assert maximum.device == hidden.device and acc.device == hidden.device
    assert maximum.shape == labels.shape == acc.shape
    assert maximum.is_cuda and acc.is_cuda

    vocab_per_split = 1024
    assert vocab_per_split % 128 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    assert entropy_b.is_contiguous() and entropy_b.is_cuda
    assert entropy_b.shape == (num_tokens,)

    if _config._backward == BackwardEnum._Total_Fuse_MN:
        # --- Triton doesn't materialize d_logits at all. Split tiles at the perspective of d_logits.
        mainloop_grid = (
            triton.cdiv(num_tokens, _config.mn_block_size_m) * triton.cdiv(vocab_size, _config.mn_block_size_n),
        )

        efficient_entropy_backward_kernel_general_mainloop_MN[mainloop_grid](
            num_tokens,
            hidden_size,
            vocab_size,
            _rank,
            hidden,
            hidden.stride(0),
            hidden.stride(1),
            weight,
            weight.stride(0),
            weight.stride(1),
            labels,
            labels.stride(0),
            maximum,
            maximum.stride(0),
            acc,
            acc.stride(0),
            dentropy,
            dentropy.stride(0),
            dlogprobs,
            dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
            REDUCTION,
            entropy_b,
            entropy_b.stride(0),
            d_hidden,
            d_hidden.stride(0),
            d_hidden.stride(1),
            d_weight,
            d_weight.stride(0),
            d_weight.stride(1),
            1.0 / temperature,
            BLOCK_SIZE_M=_config.mn_block_size_m,
            BLOCK_SIZE_N=_config.mn_block_size_n,
            BLOCK_SIZE_K=_config.mn_block_size_k,
            GROUP_SIZE_M=_config.mn_group_size_m,
            USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
            num_stages=_config.mn_num_stages,
            num_warps=_config.mn_num_warps,
        )

    elif _config._backward == BackwardEnum._Total_Separate:
        _d_logits = torch.empty((num_tokens, vocab_size), device=hidden.device, dtype=hidden.dtype).contiguous()
        assert _d_logits.is_contiguous()

        if _config._use_triton:

            d_logits_grid = (
                triton.cdiv(num_tokens, _config.dlogits_block_size_m)
                * triton.cdiv(vocab_size, _config.dlogits_block_size_n),
            )

            efficient_entropy_backward_kernel_general_d_logits[d_logits_grid](
                num_tokens,
                hidden_size,
                vocab_size,
                _rank,
                hidden,
                hidden.stride(0),
                hidden.stride(1),
                weight,
                weight.stride(0),
                weight.stride(1),
                labels,
                labels.stride(0),
                maximum,
                maximum.stride(0),
                acc,
                acc.stride(0),
                dentropy,
                dentropy.stride(0),
                dlogprobs,
                dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
                REDUCTION,
                entropy_b,
                entropy_b.stride(0),
                _d_logits,
                _d_logits.stride(0),
                _d_logits.stride(1),
                1.0 / temperature,
                BLOCK_SIZE_M=_config.dlogits_block_size_m,
                BLOCK_SIZE_N=_config.dlogits_block_size_n,
                BLOCK_SIZE_K=_config.dlogits_block_size_k,
                GROUP_SIZE_M=_config.dlogits_group_size_m,
                USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
                num_stages=_config.dlogits_num_stages,
                num_warps=_config.dlogits_num_warps,
            )

            torch.matmul(_d_logits, weight, out=d_hidden)
            torch.matmul(_d_logits.T, hidden, out=d_weight)
        else:
            raise AssertionError("Triton is required for efficient entropy kernel")

    elif _config._backward == BackwardEnum._Split_Dlogits_N:
        vocab_per_split = _config.split_n_vocab_per_split
        num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

        _d_logits = torch.empty((num_tokens, vocab_per_split), device=hidden.device, dtype=hidden.dtype).contiguous()
        assert _d_logits.is_contiguous()

        d_logits_grid = (
            triton.cdiv(num_tokens, _config.dlogits_block_size_m)
            * triton.cdiv(vocab_per_split, _config.dlogits_block_size_n),
        )

        for split_idx in range(num_splits):
            efficient_entropy_backward_kernel_general_d_logits_split_N[d_logits_grid](
                split_idx,
                num_tokens,
                hidden_size,
                vocab_size,
                vocab_per_split,
                _rank,
                hidden,
                hidden.stride(0),
                hidden.stride(1),
                weight,
                weight.stride(0),
                weight.stride(1),
                labels,
                labels.stride(0),
                maximum,
                maximum.stride(0),
                acc,
                acc.stride(0),
                dentropy,
                dentropy.stride(0),
                dlogprobs,
                dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
                REDUCTION,
                entropy_b,
                entropy_b.stride(0),
                _d_logits,
                _d_logits.stride(0),
                _d_logits.stride(1),
                1.0 / temperature,
                BLOCK_SIZE_M=_config.dlogits_block_size_m,
                BLOCK_SIZE_N=_config.dlogits_block_size_n,
                BLOCK_SIZE_K=_config.dlogits_block_size_k,
                GROUP_SIZE_M=_config.dlogits_group_size_m,
                USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
                num_stages=_config.dlogits_num_stages,
                num_warps=_config.dlogits_num_warps,
            )

            if split_idx == (num_splits - 1):
                vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size) - split_idx * vocab_per_split
                _d_logits = _d_logits[:, :vocab_right_bound].contiguous()

            if split_idx == 0:
                torch.matmul(
                    _d_logits, weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :], out=d_hidden
                )
            else:
                d_hidden += torch.matmul(
                    _d_logits, weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :]
                )
            torch.matmul(
                _d_logits.T, hidden, out=d_weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :]
            )

    elif _config._backward == BackwardEnum._Split_Dlogits_M:
        raise NotImplementedError("BackwardEnum._Split_Dlogits_M is not implemented yet")

    elif _config._backward == BackwardEnum._Split_Fused_N:
        vocab_per_split = _config.fused_n_vocab_per_split
        num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    
        # 不需要分配 d_logits 了！
        # 直接分配 d_hidden 和 d_weight（已在函数开头分配）

        d_logits_grid = (
            triton.cdiv(num_tokens, _config.backward_block_size_m)
            * triton.cdiv(vocab_per_split, _config.backward_block_size_n),
        )

        for split_idx in range(num_splits):
            efficient_entropy_backward_kernel_fused_split_n[d_logits_grid](
                split_idx,
                num_tokens,
                hidden_size,
                vocab_size,
                vocab_per_split,
                _rank,
                hidden,
                hidden.stride(0),
                hidden.stride(1),
                weight,
                weight.stride(0),
                weight.stride(1),
                labels,
                labels.stride(0),
                maximum,
                maximum.stride(0),
                acc,
                acc.stride(0),
                dentropy,
                dentropy.stride(0),
                dlogprobs,
                dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
                REDUCTION,
                entropy_b,
                entropy_b.stride(0),
                d_hidden,
                d_hidden.stride(0),
                d_hidden.stride(1),
                d_weight,
                d_weight.stride(0),
                d_weight.stride(1),
                1.0 / temperature,
                BLOCK_SIZE_M=_config.backward_block_size_m,
                BLOCK_SIZE_N=_config.backward_block_size_n,
                BLOCK_SIZE_K=_config.backward_block_size_k,
                GROUP_SIZE_M=_config.backward_group_size_m,
                USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
                num_stages=_config.backward_num_stages,
                num_warps=_config.backward_num_warps,
            )

    elif _config._backward == BackwardEnum._Split_Fused_N_Grouped:
        vocab_per_split = _config.fused_n_vocab_per_split
        N_GROUP = _config.grouped_n_group
        grouped_vocab = vocab_per_split * N_GROUP
        num_groups = (vocab_size + grouped_vocab - 1) // grouped_vocab

        grid_fn = (
            triton.cdiv(num_tokens, _config.backward_block_size_m)
            * triton.cdiv(vocab_per_split, _config.backward_block_size_n),
        )

        for group_idx in range(num_groups):
            efficient_entropy_backward_kernel_fused_split_n_grouped[grid_fn](
                group_idx,
                N_GROUP,
                num_tokens,
                hidden_size,
                vocab_size,
                vocab_per_split,
                _rank,
                hidden,
                hidden.stride(0),
                hidden.stride(1),
                weight,
                weight.stride(0),
                weight.stride(1),
                labels,
                labels.stride(0),
                maximum,
                maximum.stride(0),
                acc,
                acc.stride(0),
                dentropy,
                dentropy.stride(0),
                dlogprobs,
                dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
                REDUCTION,
                entropy_b,
                entropy_b.stride(0),
                d_hidden,
                d_hidden.stride(0),
                d_hidden.stride(1),
                d_weight,
                d_weight.stride(0),
                d_weight.stride(1),
                1.0 / temperature,
                BLOCK_SIZE_M=_config.backward_block_size_m,
                BLOCK_SIZE_N=_config.backward_block_size_n,
                BLOCK_SIZE_K=_config.backward_block_size_k,
                GROUP_SIZE_M=_config.backward_group_size_m,
                USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
                num_stages=_config.backward_num_stages,
                num_warps=_config.backward_num_warps,
            )

    elif _config._backward == BackwardEnum._Owner_Compute:

        N_GROUP = _config.owner_compute_n_group
        BLOCK_SIZE_N = _config.owner_compute_block_size_n
    
        # 计算 N-split 总数和 N-group 数量
        num_n_splits = triton.cdiv(vocab_size, BLOCK_SIZE_N)
        num_n_groups = triton.cdiv(num_n_splits, N_GROUP)

        # 关键：grid 直接覆盖所有 M 和所有 N-group
        # 一次 launch 就能处理全部工作
        grid_fn = (triton.cdiv(num_tokens, _config.backward_block_size_m) * num_n_groups,)

        efficient_entropy_backward_kernel_owner_compute[grid_fn](
            0,
            N_GROUP,
            num_tokens,
            hidden_size,
            vocab_size,
            _rank,
            hidden,
            hidden.stride(0),
            hidden.stride(1),
            weight,
            weight.stride(0),
            weight.stride(1),
            labels,
            labels.stride(0),
            maximum,
            maximum.stride(0),
            acc,
            acc.stride(0),
            dentropy,
            dentropy.stride(0),
            dlogprobs,
            dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
            REDUCTION,
            entropy_b,
            entropy_b.stride(0),
            d_hidden,
            d_hidden.stride(0),
            d_hidden.stride(1),
            d_weight,
            d_weight.stride(0),
            d_weight.stride(1),
            1.0 / temperature,
            BLOCK_SIZE_M=_config.backward_block_size_m,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=_config.backward_block_size_k,
            GROUP_SIZE_M=_config.backward_group_size_m,
            USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
            num_stages=_config.backward_num_stages,
            num_warps=_config.backward_num_warps,
        )

    return d_hidden, d_weight
