#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize matched single-node FFD and repack Kineto replays."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from pathlib import Path

from summarize_comm_overlap import RANK_PATTERN, summarize_overlap
from summarize_comm_trace import GPU_CATEGORIES, nccl_collective, resolve_traces

ARMS = ("ffd", "repack")
OPS = ("all_gather", "reduce_scatter")


def rank_traces(directory: Path) -> dict[int, Path]:
    traces = {}
    for path in resolve_traces([directory]):
        match = RANK_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"{path}: missing profiler rank in file name")
        rank = int(match[1])
        if rank in traces:
            raise ValueError(f"{directory}: multiple traces for rank {rank}")
        traces[rank] = path
    if set(traces) != set(range(4)):
        raise ValueError(f"{directory}: expected traces for ranks 0 through 3")
    return dict(sorted(traces.items()))


def collective_starts(path: Path) -> dict[str, list[tuple[str, int]]] | None:
    """Use integer nanoseconds to preserve precision with a large time origin."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    origin = document.get("baseTimeNanoseconds") if isinstance(document, dict) else None
    if type(origin) is not int or origin <= 0:
        return None

    starts = {operation: [] for operation in OPS}
    streams = {operation: set() for operation in OPS}
    for event in document["traceEvents"]:
        if event.get("ph") != "X" or event.get("cat") not in GPU_CATEGORIES:
            continue
        name = event["name"]
        if "nccl" not in name.lower() or event["dur"] == 0:
            continue
        operation = nccl_collective(name)
        if operation in starts:
            stream_id = (event.get("args") or {}).get("stream")
            if type(stream_id) is not int:
                return None
            streams[operation].add(stream_id)
            starts[operation].append((name, origin + round(event["ts"] * 1_000)))
    return {
        operation: sorted(starts[operation], key=lambda row: row[1])
        for operation in OPS
        if len(streams[operation]) == 1
    }


def matched_start_spreads(
    arms: dict[str, list[dict[str, list[tuple[str, int]]] | None]], operation: str
) -> dict[str, list[float]] | None:
    """Pair a kernel by its ordinal only when all traces have the same single-stream name sequence."""
    signature = None
    spreads = {}
    for arm, ranks in arms.items():
        by_rank = []
        for rank in ranks:
            series = rank.get(operation) if rank is not None else None
            if not series:
                return None
            names, timestamps = zip(*series)
            if signature is None:
                signature = names
            if names != signature:
                return None
            by_rank.append(timestamps)
        spreads[arm] = [(max(group) - min(group)) / 1e6 for group in zip(*by_rank, strict=True)]
    return spreads


def report(directories: tuple[Path, Path]) -> None:
    traces = {arm: rank_traces(path) for arm, path in zip(ARMS, directories, strict=True)}
    overlaps = {arm: {rank: summarize_overlap(path) for rank, path in paths.items()} for arm, paths in traces.items()}
    starts = {arm: [collective_starts(path) for path in paths.values()] for arm, paths in traces.items()}

    print("# Single-step four-rank trace")
    print()
    print("| Policy | Rank | GPU kernel span ms | NCCL active ms | NCCL overlap ms | NCCL only ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for arm, ranks in overlaps.items():
        for rank, row in ranks.items():
            print(f"| {arm} | {rank} | {row.gpu_span / 1e3:.3f} | {row.nccl / 1e3:.3f} | "
                  f"{row.both / 1e3:.3f} | {row.nccl_only / 1e3:.3f} |")

    print("\n| Operation | Policy | Matched kernels | Rank GPU start spread P50 ms | P95 ms |")
    print("|---|---|---:|---:|---:|")
    for operation in OPS:
        matched = matched_start_spreads(starts, operation)
        if matched is None:
            print(f"| {operation} | unavailable | — | — | — |")
            continue
        for arm, values in matched.items():
            p95 = sorted(values)[math.ceil(0.95 * len(values)) - 1]
            print(f"| {operation} | {arm} | {len(values)} | {statistics.median(values):.3f} | {p95:.3f} |")
    print("\nSpread uses each trace's Kineto clock origin and matches ordinal kernel names on a single stream per operation.")
    print("It describes GPU kernel start skew. Host readiness, peer wait, and the training critical path require other evidence.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ffd", type=Path, help="directory containing the four FFD trace files")
    parser.add_argument("repack", type=Path, help="directory containing the four repack trace files")
    args = parser.parse_args()
    try:
        report((args.ffd, args.repack))
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
