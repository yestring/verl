#!/usr/bin/env python3
"""
benchmark_fwd.py

Benchmark the forward pass of the efficient entropy kernel:
- Baseline: original Mainloop + Epilogue (two kernels, intermediate tensors)
- Fused: new fused kernel (single kernel, no intermediate tensors)

Measures:
- Peak GPU memory allocation (in MB)
- Kernel execution time (in ms) averaged over multiple warmup runs

Usage:
    python benchmark_fwd.py [--num-tokens N] [--hidden-size H] [--vocab-size V] [--reduction R] [--trials T]
"""

import argparse

import torch

# Assuming the kernels.py is in the same directory or PYTHONPATH
from kernels import _config, efficient_entropy_forward

# Check if fused kernel is defined (we need to import it; if not, we define a dummy)
try:
    from kernels import efficient_entropy_kernel_fused_forward  # noqa: F401 (availability check)

    FUSED_AVAILABLE = True
except ImportError:
    FUSED_AVAILABLE = False
    print("Warning: fused forward kernel not found. Only baseline will be run.")


def benchmark_forward(
    hidden, weight, labels, reduction=2, temperature=1.0, use_fused=False, num_warmup=5, num_iters=10
):
    """
    Run the forward pass and measure time and peak memory.
    Returns: (time_ms, peak_memory_mb, output_logprobs)
    """
    # Set the forward mode
    original_fused = _config._forward_fused
    if use_fused:
        assert FUSED_AVAILABLE, "Fused kernel not available"
        _config._forward_fused = True
    else:
        _config._forward_fused = False

    # Synchronize before starting
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(num_warmup):
        _ = efficient_entropy_forward(
            hidden, weight, labels,
            reduction=reduction,
            temperature=temperature,
            dist_process_group=None
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(num_iters):
        start_event.record()
        out = efficient_entropy_forward(
            hidden, weight, labels,
            reduction=reduction,
            temperature=temperature,
            dist_process_group=None
        )
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    # Peak memory after the last run
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    # Restore original setting
    _config._forward_fused = original_fused

    avg_time = sum(times) / len(times)
    return avg_time, peak_memory, out


def main():
    parser = argparse.ArgumentParser(description="Benchmark forward entropy kernel")
    parser.add_argument("--num-tokens", type=int, default=4096, help="Number of tokens (M)")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden size (K)")
    parser.add_argument("--vocab-size", type=int, default=128000, help="Vocabulary size (N)")
    parser.add_argument("--reduction", type=int, default=2, choices=[0, 1, 2],
                        help="Reduction: 0=none, 1=sum, 2=mean")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling")
    parser.add_argument("--trials", type=int, default=10, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Generate random inputs
    hidden = torch.randn(args.num_tokens, args.hidden_size, device=device, dtype=torch.float32)
    weight = torch.randn(args.vocab_size, args.hidden_size, device=device, dtype=torch.float32)
    labels = torch.randint(0, args.vocab_size, (args.num_tokens,), device=device)

    print(f"Benchmarking with M={args.num_tokens}, K={args.hidden_size}, N={args.vocab_size}")
    print(f"Reduction: {args.reduction}, Temperature: {args.temperature}")
    print("-" * 60)

    # Baseline (non-fused)
    if args.trials > 0:
        time_bl, mem_bl, out_bl = benchmark_forward(
            hidden, weight, labels,
            reduction=args.reduction,
            temperature=args.temperature,
            use_fused=False,
            num_warmup=args.warmup,
            num_iters=args.trials
        )
        print("Baseline (Mainloop+Epilogue):")
        print(f"  Time: {time_bl:.3f} ms")
        print(f"  Peak memory: {mem_bl:.2f} MB")

    # Fused (if available)
    if FUSED_AVAILABLE:
        time_fu, mem_fu, out_fu = benchmark_forward(
            hidden, weight, labels,
            reduction=args.reduction,
            temperature=args.temperature,
            use_fused=True,
            num_warmup=args.warmup,
            num_iters=args.trials
        )
        print("Fused Forward:")
        print(f"  Time: {time_fu:.3f} ms")
        print(f"  Peak memory: {mem_fu:.2f} MB")

        # Compare outputs
        if args.reduction == 0:
            # Vectors
            diff = torch.abs(out_bl[0] - out_fu[0]).max().item()
            print(f"  Max logprobs diff: {diff:.6e}")
        else:
            # Scalars
            diff = abs(out_bl[0].item() - out_fu[0].item())
            print(f"  logprobs scalar diff: {diff:.6e}")

        # Also compare entropy, maximum, accu, entropy_b if needed
        # entropy, max, accu, entropy_b are out_bl[1], out_bl[2], out_bl[3], out_bl[4]
        # We can check them for numerical equivalence
        for name, b, f in zip(
            ["entropy", "maximum", "accu", "entropy_b"],
            [out_bl[1], out_bl[2], out_bl[3], out_bl[4]],
            [out_fu[1], out_fu[2], out_fu[3], out_fu[4]],
            strict=False,
        ):
            if isinstance(b, torch.Tensor):
                maxdiff = torch.abs(b - f).max().item()
            else:
                maxdiff = abs(b - f)
            print(f"  {name} max diff: {maxdiff:.6e}")
    else:
        print("Fused kernel not available; skipping.")

    print("-" * 60)


if __name__ == "__main__":
    main()