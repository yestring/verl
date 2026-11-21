#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure per-rank overlap of NCCL and other GPU kernels in Kineto replays."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from summarize_comm_trace import GPU_CATEGORIES, resolve_traces

POLICIES = ("baseline", "defer-grad", "retain-params", "combined")
RANK_PATTERN = re.compile(r"(?:^|_)prof_rank-(\d+)_")
Span = tuple[float, float]


@dataclass(frozen=True)
class KernelOverlap:
    device: int
    gpu_span: float
    nccl: float
    other: float
    both: float
    nccl_only: float
    other_only: float
    neither: float


def resolve_rank_traces(root: Path, ranks: list[int]) -> dict[str, dict[int, Path]]:
    if not ranks or min(ranks) < 0 or len(ranks) != len(set(ranks)):
        raise ValueError("expected distinct nonnegative rank IDs")

    arms = {}
    for policy in POLICIES:
        trace_dir = root / f"replay-mixed-{policy}" / "trace"
        if not trace_dir.is_dir():
            raise ValueError(f"missing trace directory {trace_dir}")
        traces = {}
        for path in resolve_traces([trace_dir]):
            match = RANK_PATTERN.search(path.name)
            if match is None or (rank := int(match[1])) not in ranks:
                continue
            if rank in traces:
                raise ValueError(f"{trace_dir}: more than one trace for rank {rank}")
            traces[rank] = path
        if missing := sorted(set(ranks) - traces.keys()):
            raise ValueError(f"{trace_dir}: missing ranks {missing}")
        arms[policy] = traces
    return arms


def read_trace_events(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError(f"{path}: expected a Chrome trace containing event objects")
    return events


def merge(intervals: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def summarize_overlap(path: Path) -> KernelOverlap:
    devices: set[int] = set()
    nccl_kernels: list[Span] = []
    other_kernels: list[Span] = []
    for event in read_trace_events(path):
        category = event.get("cat")
        if event.get("ph") != "X" or not isinstance(category, str) or category not in GPU_CATEGORIES:
            continue
        start, duration, name = event.get("ts"), event.get("dur"), event.get("name")
        if (
            type(start) not in (int, float)
            or type(duration) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(start + duration)
            or duration < 0
            or not isinstance(name, str)
            or not name
        ):
            raise ValueError(f"{path}: malformed GPU kernel event")
        if duration == 0:
            continue
        args = event.get("args")
        device = args.get("device") if isinstance(args, dict) else None
        if type(device) is not int or device < 0:
            raise ValueError(f"{path}: missing or invalid GPU kernel device ID")
        devices.add(device)
        kernels = nccl_kernels if "nccl" in name.lower() else other_kernels
        kernels.append((start, start + duration))

    if len(devices) != 1:
        raise ValueError(f"{path}: expected kernels on one device, found {sorted(devices)}")
    if not nccl_kernels or not other_kernels:
        raise ValueError(f"{path}: expected both NCCL and non-NCCL GPU kernels")

    nccl_spans, other_spans = merge(nccl_kernels), merge(other_kernels)
    busy_spans = merge(nccl_spans + other_spans)
    nccl = math.fsum(end - start for start, end in nccl_spans)
    other = math.fsum(end - start for start, end in other_spans)
    busy = math.fsum(end - start for start, end in busy_spans)
    gpu_span = busy_spans[-1][1] - busy_spans[0][0]
    both = max(0, nccl + other - busy)
    return KernelOverlap(
        device=devices.pop(),
        gpu_span=gpu_span,
        nccl=nccl,
        other=other,
        both=both,
        nccl_only=max(0, busy - other),
        other_only=max(0, busy - nccl),
        neither=max(0, gpu_span - busy),
    )


def print_table(results: dict[str, dict[int, KernelOverlap]], ranks: list[int]) -> None:
    print(
        "| Policy | Rank | Device | GPU kernel span ms | NCCL active ms | Other kernel active ms | "
        "Overlap ms | NCCL only ms | Other only ms | No kernel ms | NCCL-only Δ vs baseline ms |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy, by_rank in results.items():
        for rank in ranks:
            row = by_rank[rank]
            delta = row.nccl_only - results["baseline"][rank].nccl_only
            print(
                f"| {policy} | {rank} | {row.device} | {row.gpu_span / 1000:.3f} | {row.nccl / 1000:.3f} | "
                f"{row.other / 1000:.3f} | {row.both / 1000:.3f} | {row.nccl_only / 1000:.3f} | "
                f"{row.other_only / 1000:.3f} | {row.neither / 1000:.3f} | {delta / 1000:+.3f} |"
            )

    print("\nEach rank uses its own first-to-last GPU kernel span. Active times merge intervals across streams.")
    print("Other means non-NCCL CUDA kernels. Copies and memsets are outside the kernel categories.")
    print("NCCL-only time describes observed overlap. Critical-path time and step speedup require separate evidence.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path, help="root directory produced by run_comm_trace.sh")
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    try:
        traces = resolve_rank_traces(args.trace_root, args.ranks)
        results = {
            policy: {rank: summarize_overlap(path) for rank, path in by_rank.items()}
            for policy, by_rank in traces.items()
        }
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print_table(results, args.ranks)


if __name__ == "__main__":
    main()
