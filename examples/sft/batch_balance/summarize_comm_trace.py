#!/usr/bin/env python3
"""Summarize CPU collective dispatches and NCCL kernels from Kineto traces."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

COLLECTIVES = ("reduce_scatter", "all_gather", "all_reduce")
COMPACT_NAMES = {name.replace("_", ""): name for name in COLLECTIVES}
CPU_PREFIXES = ("c10d::", "_c10d_functional::", "c10d_functional::", "nccl:")
CPU_OP_SUFFIXES = {
    "", "base", "basecoalesced", "coalesced", "tensor", "tensorout",
    "tensorcoalesced", "intotensor", "intotensorout", "intotensorcoalesced",
}
CPU_CATEGORIES = {"cpu_op", "user_annotation", "cpu_user_annotation"}
GPU_CATEGORIES = {"kernel", "gpu_kernel"}


def cpu_collective(name: str) -> str | None:
    for prefix in CPU_PREFIXES:
        if name.startswith(prefix):
            op = name[len(prefix) :].replace("_", "").lower()
            for compact, collective in COMPACT_NAMES.items():
                if op.startswith(compact) and op[len(compact) :] in CPU_OP_SUFFIXES:
                    return collective
            return None
    return None


def nccl_collective(name: str) -> str:
    compact = name.lower().replace("_", "")
    for token, collective in COMPACT_NAMES.items():
        if token in compact:
            return collective
    return "other_nccl"


def count_dispatches(intervals: list[tuple[float, float]]) -> int:
    """Count innermost scopes of one operation and host thread.

    A frontend c10d range can contain one or more backend NCCL ranges. Keep
    distinct inner ranges and collapse same-interval duplicates.
    """
    earliest_end = math.inf
    count = 0
    for _, end in sorted(intervals, key=lambda span: (-span[0], span[1])):
        if end < earliest_end:
            count += 1
            earliest_end = end
    return count


def summarize(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        trace = json.load(f)
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(events, list):
        raise ValueError("expected a Chrome trace containing traceEvents")

    host_intervals = defaultdict(list)
    cpu_events = 0
    gpu_kernels = 0
    result = {
        name: {"cpu_dispatches": 0, "gpu_kernels": 0, "gpu_kernel_ms": 0.0}
        for name in (*COLLECTIVES, "other_nccl")
    }
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = event.get("name", "")
        category = event.get("cat", "")
        duration = event.get("dur")
        timestamp = event.get("ts")
        if not isinstance(name, str) or not isinstance(category, str):
            continue
        if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 0:
            continue
        if category in GPU_CATEGORIES:
            gpu_kernels += 1
            if "nccl" in name.lower():
                counters = result[nccl_collective(name)]
                counters["gpu_kernels"] += 1
                counters["gpu_kernel_ms"] += duration / 1000.0
        elif category in CPU_CATEGORIES:
            cpu_events += 1
            collective = cpu_collective(name)
            if collective and isinstance(timestamp, (float, int)) and math.isfinite(timestamp):
                thread = (event.get("pid"), event.get("tid"), collective)
                host_intervals[thread].append((timestamp, timestamp + duration))

    for (_, _, collective), intervals in host_intervals.items():
        result[collective]["cpu_dispatches"] += count_dispatches(intervals)
    return {
        "trace": str(path),
        "cpu_event_count": cpu_events,
        "gpu_kernel_count": gpu_kernels,
        "collectives": result,
    }


def resolve_traces(paths: list[Path]) -> list[Path]:
    traces = []
    for path in paths:
        if path.is_dir():
            matches = sorted(set(path.rglob("*prof_rank-*.json")) | set(path.rglob("*prof_rank-*.json.gz")))
            if not matches:
                raise ValueError(f"no verl TorchProfiler traces in {path}")
            traces.extend(matches)
        else:
            traces.append(path)
    return list(dict.fromkeys(traces))


def print_table(rows: list[dict]) -> None:
    print("| Trace | CPU events | CUDA kernels | Operation | CPU dispatches | NCCL CUDA kernels | Kernel sum ms |")
    print("|---|---:|---:|---|---:|---:|---:|")
    for row in rows:
        cpu_events, gpu_kernels = row["cpu_event_count"], row["gpu_kernel_count"]
        trace = row["trace"].replace("|", "\\|")
        for name, stats in row["collectives"].items():
            cpu = stats["cpu_dispatches"] if cpu_events and name != "other_nccl" else "–"
            kernels = stats["gpu_kernels"] if gpu_kernels else "–"
            kernel_ms = f"{stats['gpu_kernel_ms']:.3f}" if gpu_kernels else "–"
            print(f"| `{trace}` | {cpu_events} | {gpu_kernels} | {name} | {cpu} | {kernels} | {kernel_ms} |")
    print("\nCPU counts use recognized dispatch scopes. CUDA counts use NCCL kernel names.")
    print("A zero means no matching event names. A dash marks missing coverage or an unavailable classification.")
    print("Kernel duration sums include overlap.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path, help="Chrome trace (.json or .json.gz) or artifacts directory")
    parser.add_argument("--json", action="store_true", help="emit machine-readable per-trace measurements")
    args = parser.parse_args()
    try:
        rows = [summarize(path) for path in resolve_traces(args.trace)]
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
