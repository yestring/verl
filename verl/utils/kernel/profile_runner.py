#!/usr/bin/env python3
"""
VeRL Entropy Kernel 前后向 Profiling Runner
用于 Nsight Systems / Nsight Compute 性能分析
"""

import torch
import triton
import sys
import os

# 将 verl 的 kernel 路径加入 sys.path（根据你的实际路径调整）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import (
    efficient_entropy_forward,
    efficient_entropy_backward,
    _config,
    BackwardEnum,
)


def run_forward_only(M, K, V, iters=100, warmup=10):
    """仅跑前向，用于对比基准"""
    hidden = torch.randn(M, K, device='cuda', dtype=torch.float32)
    weight = torch.randn(V, K, device='cuda', dtype=torch.float32)
    labels = torch.randint(0, V, (M,), device='cuda')

    # 预热
    for _ in range(warmup):
        logprobs, entropy, max_, accu, entropy_b = efficient_entropy_forward(
            hidden, weight, labels, reduction=2, temperature=1.0
        )
    torch.cuda.synchronize()

    # 正式迭代
    for _ in range(iters):
        logprobs, entropy, max_, accu, entropy_b = efficient_entropy_forward(
            hidden, weight, labels, reduction=2, temperature=1.0
        )
    torch.cuda.synchronize()


def run_backward_only(M, K, V, strategy: BackwardEnum, iters=100, warmup=10):
    """
    跑反向传播（需先跑前向获得必要的统计量）
    strategy: BackwardEnum._Total_Fuse_MN / _Total_Separate / _Split_Dlogits_N
    """
    # 设置反向策略
    _config._backward = strategy

    hidden = torch.randn(M, K, device='cuda', dtype=torch.float32)
    weight = torch.randn(V, K, device='cuda', dtype=torch.float32)
    labels = torch.randint(0, V, (M,), device='cuda')

    # ---- 前向：准备反向所需的数据 ----
    logprobs, entropy, max_, accu, entropy_b = efficient_entropy_forward(
        hidden, weight, labels, reduction=2, temperature=1.0
    )

    # 构造 dlogprobs 和 dentropy（模拟 VeRL 传回的梯度）
    # reduction=2 表示 mean，所以 dlogprobs 是标量
    dlogprobs = torch.ones((), device='cuda', dtype=torch.float32)
    dentropy = torch.ones((M,), device='cuda', dtype=torch.float32)

    # 预热
    for _ in range(warmup):
        d_hidden, d_weight = efficient_entropy_backward(
            dlogprobs=dlogprobs,
            dentropy=dentropy,
            hidden=hidden,
            weight=weight,
            labels=labels,
            maximum=max_,
            acc=accu,
            entropy_b=entropy_b,
            reduction=2,
            should_return_fp32_grad=True,
            temperature=1.0,
        )
    torch.cuda.synchronize()

    # 正式迭代
    for _ in range(iters):
        d_hidden, d_weight = efficient_entropy_backward(
            dlogprobs=dlogprobs,
            dentropy=dentropy,
            hidden=hidden,
            weight=weight,
            labels=labels,
            maximum=max_,
            acc=accu,
            entropy_b=entropy_b,
            reduction=2,
            should_return_fp32_grad=True,
            temperature=1.0,
        )
    torch.cuda.synchronize()


def run_forward_backward(M, K, V, strategy: BackwardEnum, iters=100, warmup=10):
    """
    完整的前向+反向（模拟真实训练迭代）
    """
    _config._backward = strategy

    hidden = torch.randn(M, K, device='cuda', dtype=torch.float32)
    weight = torch.randn(V, K, device='cuda', dtype=torch.float32)
    labels = torch.randint(0, V, (M,), device='cuda')

    # 预热
    for _ in range(warmup):
        logprobs, entropy, max_, accu, entropy_b = efficient_entropy_forward(
            hidden, weight, labels, reduction=2, temperature=1.0
        )
        dlogprobs = torch.ones((), device='cuda', dtype=torch.float32)
        dentropy = torch.ones((M,), device='cuda', dtype=torch.float32)
        d_hidden, d_weight = efficient_entropy_backward(
            dlogprobs=dlogprobs,
            dentropy=dentropy,
            hidden=hidden,
            weight=weight,
            labels=labels,
            maximum=max_,
            acc=accu,
            entropy_b=entropy_b,
            reduction=2,
            should_return_fp32_grad=True,
            temperature=1.0,
        )
    torch.cuda.synchronize()

    # 正式迭代
    for _ in range(iters):
        logprobs, entropy, max_, accu, entropy_b = efficient_entropy_forward(
            hidden, weight, labels, reduction=2, temperature=1.0
        )
        dlogprobs = torch.ones((), device='cuda', dtype=torch.float32)
        dentropy = torch.ones((M,), device='cuda', dtype=torch.float32)
        d_hidden, d_weight = efficient_entropy_backward(
            dlogprobs=dlogprobs,
            dentropy=dentropy,
            hidden=hidden,
            weight=weight,
            labels=labels,
            maximum=max_,
            acc=accu,
            entropy_b=entropy_b,
            reduction=2,
            should_return_fp32_grad=True,
            temperature=1.0,
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    # ============ 配置区 ============
    # 使用与之前一致的 workload（可根据需要调整）
    M = 1024   # num_tokens
    K = 1024   # hidden_size
    V = 128256//4 # vocab_size

    # 选择要运行的测试（取消注释对应的行）
    # -------------------------------------------------
    # 1. 仅前向（baseline）
    # run_forward_only(M, K, V, iters=100, warmup=10)

    # 2. 三种反向策略单独测试
    # run_backward_only(M, K, V, BackwardEnum._Total_Fuse_MN, iters=100, warmup=10)
    # run_backward_only(M, K, V, BackwardEnum._Total_Separate, iters=100, warmup=10)
    # run_backward_only(M, K, V, BackwardEnum._Split_Dlogits_N, iters=100, warmup=10)

    # 3. 完整前向+反向（模拟真实迭代）
    run_forward_backward(M, K, V, BackwardEnum._Owner_Compute, iters=1, warmup=0)
    # -------------------------------------------------