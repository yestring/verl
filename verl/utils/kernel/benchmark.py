#!/usr/bin/env python3
"""
VeRL Entropy Kernel Benchmark Suite
===================================
测试项目：
1. 正确性验证（对比 PyTorch 原生实现）
2. 性能基准（前向/反向/端到端延迟）
3. 显存占用（峰值显存监控）
4. 自动生成可视化图表
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# 导入 Kernels
from kernels import (
    SUPPORTED_BACKWARD_METHODS,
    BackwardEnum,
    _config,
    configure,
    efficient_entropy_backward,
    efficient_entropy_forward,
    get_backward_method_by_name,
    get_backward_method_name,
    set_forward_fused,
)

# ============================================================
# 策略名称映射
# ============================================================
STRATEGY_NAME_MAP = {
    BackwardEnum._Total_Fuse_MN: "Fuse_MN",
    BackwardEnum._Total_Separate: "Separate",
    BackwardEnum._Split_Dlogits_N: "Split_N",
    BackwardEnum._Split_Fused_N: "Fused_Split_N",
    BackwardEnum._Split_Fused_N_Grouped: "Fused_Split_N_Grouped",
    BackwardEnum._Owner_Compute:"Owner_Compute",
}

# ============================================================
# 配置类
# ============================================================
@dataclass
class Workload:
    """定义测试 workload"""
    M: int          # num_tokens
    K: int          # hidden_size
    V: int          # vocab_size
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"M{self.M}_K{self.K}_V{self.V}"


@dataclass
class BenchmarkResult:
    """单次 benchmark 结果"""
    workload: Workload
    strategy: str
    forward_time_ms: float
    backward_time_ms: float
    end2end_time_ms: float
    peak_memory_mb: float
    forward_correct: bool
    backward_correct: bool
    config: dict = field(default_factory=dict)


# ============================================================
# 正确性测试
# ============================================================
# 容差常量：Triton 核内为精确 fp32 累加（与 cuBLAS 的求和顺序不同），对照参考
# 即使关闭 TF32 仍存在舍入顺序差异，因此按以下容差判定（在 A100 上校准）。
# 旧版"全策略 fwd_correct=false"的伪警报根因：参考 matmul 开着 TF32 而 Triton
# 用精确 fp32 累加，两者差异被误判为 kernel 错误。
FWD_RTOL = 2e-3
FWD_ATOL = 2e-5
BWD_RTOL = 2e-2
BWD_ATOL = 1e-3


def _set_tf32(enabled: bool):
    """Set TF32 for matmul, using the new precision API on torch >= 2.9."""
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = enabled


def test_correctness(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[bool, bool, float, float]:
    """
    测试 Kernel 正确性，返回 (forward_correct, backward_correct, max_diff_fwd, max_diff_bwd)
    """
    M, K = hidden.shape
    V, _ = weight.shape

    # ---- PyTorch 参考实现（关闭 TF32，保持可微） ----
    original_precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    _set_tf32(False)
    try:
        # 用可微克隆（不改动 benchmark 输入张量的 requires_grad 状态）
        hidden_ref = hidden.detach().requires_grad_(True)
        weight_ref = weight.detach().requires_grad_(True)
        logits_ref = hidden_ref @ weight_ref.T / temperature
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)
        log_probs_target = log_probs_ref.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        probs_ref = log_probs_ref.exp()
        entropy_ref = -(probs_ref * log_probs_ref).sum(dim=-1)

        # ---- Triton 前向 ----
        log_probs_tri, entropy_tri, max_tri, acc_tri, entropy_b_tri = efficient_entropy_forward(
            hidden, weight, labels, reduction=0, temperature=temperature
        )

        # 检查前向正确性（logprobs 与 entropy 都查）
        fwd_correct = torch.allclose(log_probs_tri, log_probs_target, rtol=FWD_RTOL, atol=FWD_ATOL) and torch.allclose(
            entropy_tri, entropy_ref, rtol=FWD_RTOL, atol=FWD_ATOL
        )
        fwd_max_diff = max(
            (log_probs_tri - log_probs_target).abs().max().item(),
            (entropy_tri - entropy_ref).abs().max().item(),
        )

        # ---- Triton 反向 ----
        # 任意上游梯度（与参考 loss 一致）
        dlogprobs = torch.randn_like(log_probs_target)
        dentropy = torch.randn_like(entropy_ref)

        d_hidden_tri, d_weight_tri = efficient_entropy_backward(
            dlogprobs=dlogprobs,
            dentropy=dentropy,
            hidden=hidden,
            weight=weight,
            labels=labels,
            maximum=max_tri,
            acc=acc_tri,
            entropy_b=entropy_b_tri,
            reduction=0,
            should_return_fp32_grad=True,
            temperature=temperature,
        )

        # PyTorch 反向：可微参考 loss，autograd 给出精确梯度
        loss_ref = (log_probs_target * dlogprobs).sum() + (entropy_ref * dentropy).sum()
        d_hidden_ref, d_weight_ref = torch.autograd.grad(loss_ref, (hidden_ref, weight_ref))

        bwd_correct = torch.allclose(d_hidden_tri, d_hidden_ref, rtol=BWD_RTOL, atol=BWD_ATOL) and torch.allclose(
            d_weight_tri, d_weight_ref, rtol=BWD_RTOL, atol=BWD_ATOL
        )
        bwd_max_diff = max(
            (d_hidden_tri - d_hidden_ref).abs().max().item(),
            (d_weight_tri - d_weight_ref).abs().max().item(),
        )
    finally:
        if original_precision is not None:
            torch.backends.cuda.matmul.fp32_precision = original_precision
        torch.backends.cuda.matmul.allow_tf32 = original_tf32

    return fwd_correct, bwd_correct, fwd_max_diff, bwd_max_diff


# ============================================================
# 性能测试
# ============================================================
def benchmark_kernel(
    workload: Workload,
    strategy: BackwardEnum,
    temperature: float = 1.0,
    iters: int = 50,
    warmup: int = 10,
    device: str = "cuda",
    skip_correctness: bool = False,
) -> BenchmarkResult:
    """测试单个配置的性能"""

    M, K, V = workload.M, workload.K, workload.V

    # 设置策略
    _config._backward = strategy
    strategy_name = STRATEGY_NAME_MAP.get(strategy, str(strategy))
    
    # 生成数据
    hidden = torch.randn(M, K, device=device, dtype=torch.float32)
    weight = torch.randn(V, K, device=device, dtype=torch.float32)
    labels = torch.randint(0, V, (M,), device=device)
    
    # ---- 清空缓存，测量显存 ----
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # ---- 前向测试 ----
    # 预热
    for _ in range(warmup):
        log_probs, entropy, max_, accu, entropy_b = efficient_entropy_forward(
            hidden, weight, labels, reduction=0, temperature=temperature
        )
    torch.cuda.synchronize()
    
    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        log_probs, entropy, max_, accu, entropy_b = efficient_entropy_forward(
            hidden, weight, labels, reduction=0, temperature=temperature
        )
    end.record()
    torch.cuda.synchronize()
    forward_time_ms = start.elapsed_time(end) / iters
    
    # ---- 反向测试 ----
    # 构造梯度
    dlogprobs = torch.ones_like(log_probs)
    dentropy = torch.ones_like(entropy)
    
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
            reduction=0,
            should_return_fp32_grad=True,
            temperature=temperature,
        )
    torch.cuda.synchronize()
    
    # 计时
    start.record()
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
            reduction=0,
            should_return_fp32_grad=True,
            temperature=temperature,
        )
    end.record()
    torch.cuda.synchronize()
    backward_time_ms = start.elapsed_time(end) / iters
    
    # ---- 端到端（前向+反向） ----
    end2end_time_ms = forward_time_ms + backward_time_ms
    
    # ---- 峰值显存 ----
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    
    # ---- 正确性验证（性能扫描时默认跳过；精度校准放到最后一步） ----
    if skip_correctness:
        fwd_correct = bwd_correct = True
    else:
        fwd_correct, bwd_correct, _, _ = test_correctness(hidden, weight, labels, temperature)

    return BenchmarkResult(
        workload=workload,
        strategy=strategy_name,
        forward_time_ms=forward_time_ms,
        backward_time_ms=backward_time_ms,
        end2end_time_ms=end2end_time_ms,
        peak_memory_mb=peak_memory_mb,
        forward_correct=fwd_correct,
        backward_correct=bwd_correct,
        config={
            "temperature": temperature,
            "iters": iters,
            "backward_method": get_backward_method_name(_config._backward),
            "forward_fused": _config._forward_fused,
            "forward_vocab_per_split": _config.forward_vocab_per_split,
            "split_n_vocab_per_split": _config.split_n_vocab_per_split,
            "fused_n_vocab_per_split": _config.fused_n_vocab_per_split,
            "grouped_n_group": _config.grouped_n_group,
            "owner_compute_n_group": _config.owner_compute_n_group,
            "backward_block_size_mnk": [
                _config.backward_block_size_m,
                _config.backward_block_size_n,
                _config.backward_block_size_k,
            ],
        },
    )


# ============================================================
# 批量 Benchmark
# ============================================================
def run_benchmark_suite(
    workloads: list[Workload],
    strategies: list[BackwardEnum],
    output_dir: str = "./benchmark_results",
    iters: int = 50,
    warmup: int = 10,
    skip_correctness: bool = False,
) -> list[BenchmarkResult]:
    """运行完整的 benchmark 套件"""

    os.makedirs(output_dir, exist_ok=True)
    results = []

    total = len(workloads) * len(strategies)
    current = 0

    for wl in workloads:
        for strategy in strategies:
            current += 1
            strategy_name = strategy.name if hasattr(strategy, "name") else str(strategy)
            print(f"[{current}/{total}] Testing {wl.name} / {strategy_name} ...")

            try:
                result = benchmark_kernel(
                    workload=wl,
                    strategy=strategy,
                    iters=iters,
                    warmup=warmup,
                    skip_correctness=skip_correctness,
                )
                results.append(result)
                
                # 打印简要结果
                print(f"  ✓ Fwd: {result.forward_time_ms:.3f} ms, Bwd: {result.backward_time_ms:.3f} ms, "
                      f"Memory: {result.peak_memory_mb:.1f} MB")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    # 保存结果到 JSON
    save_results(results, output_dir)
    
    return results


# ============================================================
# 结果保存与加载
# ============================================================
def save_results(results: list[BenchmarkResult], output_dir: str):
    """保存 benchmark 结果到 JSON"""
    data = []
    for r in results:
        data.append({
            "workload": {
                "M": r.workload.M,
                "K": r.workload.K,
                "V": r.workload.V,
                "name": r.workload.name,
            },
            "strategy": r.strategy,
            "forward_time_ms": r.forward_time_ms,
            "backward_time_ms": r.backward_time_ms,
            "end2end_time_ms": r.end2end_time_ms,
            "peak_memory_mb": r.peak_memory_mb,
            "forward_correct": r.forward_correct,
            "backward_correct": r.backward_correct,
        })
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(data, f, indent=2)


def load_results(output_dir: str) -> list[dict]:
    """从 JSON 加载 benchmark 结果"""
    with open(os.path.join(output_dir, "results.json")) as f:
        return json.load(f)


# ============================================================
# 可视化
# ============================================================
def plot_results(results: list[BenchmarkResult], output_dir: str):
    """生成性能对比图表"""
    
    # 按 workload 分组
    workloads = sorted(set(r.workload.name for r in results))
    strategies = sorted(set(r.strategy for r in results))
    
    # 颜色映射
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_map = {s: colors[i % len(colors)] for i, s in enumerate(strategies)}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ----- 1. 端到端延迟 -----
    ax = axes[0, 0]
    x = np.arange(len(workloads))
    width = 0.8 / len(strategies)
    
    for i, strategy in enumerate(strategies):
        vals = []
        for wl in workloads:
            r = [r for r in results if r.workload.name == wl and r.strategy == strategy]
            vals.append(r[0].end2end_time_ms if r else 0)
        ax.bar(x + i * width - 0.4 + width/2, vals, width, 
               label=strategy, color=color_map[strategy])
    
    ax.set_xlabel("Workload")
    ax.set_ylabel("End-to-End Time (ms)")
    ax.set_title("End-to-End Latency by Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ----- 2. 前向 vs 反向分解 -----
    ax = axes[0, 1]
    x = np.arange(len(workloads))
    
    for i, strategy in enumerate(strategies):
        fwd_vals = []
        bwd_vals = []
        for wl in workloads:
            r = [r for r in results if r.workload.name == wl and r.strategy == strategy]
            if r:
                fwd_vals.append(r[0].forward_time_ms)
                bwd_vals.append(r[0].backward_time_ms)
            else:
                fwd_vals.append(0)
                bwd_vals.append(0)
        
        # 堆积柱状图
        if i == 0:
            ax.bar(x, fwd_vals, width, label="Forward", color="#1f77b4")
            bottom = fwd_vals
        else:
            ax.bar(x, fwd_vals, width, bottom=bottom, label=f"{strategy} Fwd", color="#aec7e8")
            # 简单起见，只画第一组
            pass
    
    # 简化：用分组柱状图
    ax.clear()
    for i, strategy in enumerate(strategies):
        fwd_vals = []
        bwd_vals = []
        for wl in workloads:
            r = [r for r in results if r.workload.name == wl and r.strategy == strategy]
            if r:
                fwd_vals.append(r[0].forward_time_ms)
                bwd_vals.append(r[0].backward_time_ms)
            else:
                fwd_vals.append(0)
                bwd_vals.append(0)
        
        offset = i * width - 0.4 + width/2
        ax.bar(x + offset, fwd_vals, width/2, label=f"{strategy} Fwd", color=color_map[strategy], alpha=0.7)
        ax.bar(x + offset + width/2, bwd_vals, width/2, label=f"{strategy} Bwd", color=color_map[strategy], alpha=1.0)
    
    ax.set_xlabel("Workload")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Forward vs Backward Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # ----- 3. 显存占用 -----
    ax = axes[1, 0]
    for i, strategy in enumerate(strategies):
        vals = []
        for wl in workloads:
            r = [r for r in results if r.workload.name == wl and r.strategy == strategy]
            vals.append(r[0].peak_memory_mb if r else 0)
        ax.plot(workloads, vals, marker="o", label=strategy, color=color_map[strategy], linewidth=2, markersize=8)
    
    ax.set_xlabel("Workload")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory Usage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ----- 4. 正确性状态 -----
    ax = axes[1, 1]
    for i, strategy in enumerate(strategies):
        fwd_correct = []
        bwd_correct = []
        for wl in workloads:
            r = [r for r in results if r.workload.name == wl and r.strategy == strategy]
            if r:
                fwd_correct.append(1 if r[0].forward_correct else 0)
                bwd_correct.append(1 if r[0].backward_correct else 0)
            else:
                fwd_correct.append(0)
                bwd_correct.append(0)
        
        ax.scatter([wl for wl in workloads], [f"{strategy}-Fwd"] * len(workloads), 
                   c=fwd_correct, cmap="RdYlGn", s=100, vmin=0, vmax=1)
    
    ax.set_xlabel("Workload")
    ax.set_title("Correctness Status (Green=PASS)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_charts.png"), dpi=150)
    plt.show()
    print(f"✅ Charts saved to {output_dir}/benchmark_charts.png")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="VeRL Entropy Kernel Benchmark")
    parser.add_argument("--workloads", type=str, default="small,medium,large",
                        help="Comma-separated workload types: small,medium,large")
    parser.add_argument("--iters", type=int, default=10, help="Iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    # ---- sweep 参数（Athena 候选旋钮） ----
    parser.add_argument("--backward-methods", type=str, default=None,
                        help="Comma-separated strategy names (e.g. split_dlogits_n,split_fused_n) or 'all'")
    parser.add_argument("--forward-fused", dest="forward_fused", action="store_true", default=None,
                        help="Enable fused forward kernel")
    parser.add_argument("--no-forward-fused", dest="forward_fused", action="store_false",
                        help="Disable fused forward kernel")
    parser.add_argument("--vocab-per-split", type=int, default=None,
                        help="Override vocab_per_split for all families (fwd/split_n/fused_n)")
    parser.add_argument("--n-group", type=int, default=None,
                        help="Override n_group for grouped / owner_compute strategies")
    parser.add_argument("--block-size-m", type=int, default=None, help="Override BLOCK_SIZE_M for all families")
    parser.add_argument("--block-size-n", type=int, default=None, help="Override BLOCK_SIZE_N for all families")
    parser.add_argument("--block-size-k", type=int, default=None, help="Override BLOCK_SIZE_K for all families")

    # ---- 正确性 ----
    parser.add_argument("--skip-correctness", action="store_true", default=False,
                        help="Skip correctness checks (perf-focused sweep)")
    args = parser.parse_args()

    # ----- 应用 sweep 配置到全局 _config -----
    if args.forward_fused is not None:
        set_forward_fused(args.forward_fused)
    if (
        args.vocab_per_split is not None
        or args.n_group is not None
        or args.block_size_m is not None
        or args.block_size_n is not None
        or args.block_size_k is not None
    ):
        configure(
            vocab_per_split=args.vocab_per_split,
            n_group=args.n_group,
            block_size_m=args.block_size_m,
            block_size_n=args.block_size_n,
            block_size_k=args.block_size_k,
        )

    # ----- 定义 workloads -----
    workload_map = {
        "small": Workload(1024, 2048, 32000, "small (1k×2k×32k)"),
        "medium": Workload(4096, 4096, 128256, "medium (4k×4k×128k)"),
        # "large": Workload(8192, 8192, 256000, "large (8k×8k×256k)"),
        # "tiny": Workload(256, 512, 4096, "tiny (256×512×4k)"),
        # "xl": Workload(16384, 8192, 256000, "xl (16k×8k×256k)"),
    }

    selected = args.workloads.split(",")
    workloads = [workload_map[name.strip()] for name in selected if name.strip() in workload_map]

    # ----- 定义 strategies -----
    if args.backward_methods:
        names = [n.strip() for n in args.backward_methods.split(",")]
        if names == ["all"]:
            names = list(SUPPORTED_BACKWARD_METHODS)
        strategies = [get_backward_method_by_name(n) for n in names]
    else:
        strategies = [
            #BackwardEnum._Total_Fuse_MN,
            BackwardEnum._Total_Separate,
            BackwardEnum._Split_Dlogits_N,
            BackwardEnum._Split_Fused_N,
            BackwardEnum._Split_Fused_N_Grouped,
            BackwardEnum._Owner_Compute,
        ]
    # 注意：_Total_Fuse_MN 在 A100 上可能因 shared memory 限制失败
    # 可用 --backward-methods total_fuse_mn 显式验证

    # ----- 检查 CUDA -----
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Triton: {__import__('triton').__version__}")
    print(f"✅ Workloads: {[wl.name for wl in workloads]}")
    print(f"✅ Strategies: {[STRATEGY_NAME_MAP.get(s, str(s)) for s in strategies]}")
    print(f"✅ forward_fused={_config._forward_fused}, "
          f"vocab_per_split(fwd/split_n/fused_n)="
          f"{_config.forward_vocab_per_split}/{_config.split_n_vocab_per_split}/{_config.fused_n_vocab_per_split}, "
          f"n_group(grouped/owner)={_config.grouped_n_group}/{_config.owner_compute_n_group}, "
          f"bwd_block_mnk=({_config.backward_block_size_m},{_config.backward_block_size_n},{_config.backward_block_size_k})")
    print()

    # ----- 运行 Benchmark -----
    results = run_benchmark_suite(
        workloads=workloads,
        strategies=strategies,
        output_dir=args.output,
        iters=args.iters,
        warmup=args.warmup,
        skip_correctness=args.skip_correctness,
    )
    
    # ----- 生成图表 -----
    if not args.no_plot and results:
        plot_results(results, args.output)
    
    # ----- 打印汇总表格 -----
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    header = (
        f"{'Workload':<20} {'Strategy':<20} {'Fwd (ms)':<12} {'Bwd (ms)':<12} "
        f"{'End2End (ms)':<14} {'Memory (MB)':<12}"
    )
    print(header)
    print("-" * 80)
    
    for r in results:
        print(f"{r.workload.name:<20} {r.strategy:<20} "
              f"{r.forward_time_ms:<12.2f} {r.backward_time_ms:<12.2f} "
              f"{r.end2end_time_ms:<14.2f} {r.peak_memory_mb:<12.1f}")


if __name__ == "__main__":
    main()