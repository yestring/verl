#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
"""Compare CUDA allocator samples at matched FSDP2 microbatch boundaries."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from summarize_comm_overlap import read_trace_events, resolve_rank_traces

PHASES = ("before_forward", "after_forward", "after_backward")
MARKER_PREFIX = "sft_comm_probe.memory."
MARKER = re.compile(
    rf"{re.escape(MARKER_PREFIX)}([1-9][0-9]*)\.({'|'.join(PHASES)})"
    r"\.([0-9]+)\.([0-9]+)\.([1-9][0-9]*)\.([0-9]+)"
)
MIB = 1 << 20


@dataclass(frozen=True)
class Boundary:
    timestamp: float
    allocated: int
    reserved: int
    tokens: int
    peak: int


Boundaries = dict[tuple[int, str], Boundary]
PolicySamples = dict[str, dict[int, Boundaries]]


def read_boundaries(path: Path) -> Boundaries:
    boundaries = {}
    for event in read_trace_events(path):
        name = event.get("name")
        if not isinstance(name, str) or not name.startswith(MARKER_PREFIX):
            continue
        if event.get("ph") != "X" or event.get("cat") not in {"user_annotation", "cpu_user_annotation"}:
            continue
        match = MARKER.fullmatch(name)
        timestamp = event.get("ts")
        if not match or type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError(f"{path}: invalid memory marker {name!r}")
        index, phase, allocated, reserved, tokens, peak = match.groups()
        key = int(index), phase
        if key in boundaries:
            raise ValueError(f"{path}: duplicate microbatch {index} {phase} marker")
        boundaries[key] = Boundary(timestamp, int(allocated), int(reserved), int(tokens), int(peak))

    if not boundaries:
        raise ValueError(f"{path}: no FSDP2 CUDA memory markers")
    indices = sorted({index for index, _ in boundaries})
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError(f"{path}: noncontiguous microbatch IDs {indices}")

    ordered = {}
    previous_time = -math.inf
    previous_peak = 0
    for index in indices:
        tokens = None
        for phase in PHASES:
            key = index, phase
            sample = boundaries.get(key)
            if sample is None:
                raise ValueError(f"{path}: missing microbatch {index} {phase} marker")
            if sample.timestamp < previous_time:
                raise ValueError(f"{path}: out-of-order microbatch {index} {phase} marker")
            if tokens is not None and sample.tokens != tokens:
                raise ValueError(f"{path}: input token count changes within microbatch {index}")
            if sample.peak < max(sample.allocated, previous_peak):
                raise ValueError(f"{path}: inconsistent cumulative peak at microbatch {index} {phase}")
            ordered[key] = sample
            tokens = sample.tokens
            previous_time = sample.timestamp
            previous_peak = sample.peak
    return ordered


def read_matched_boundaries(trace_root: Path, ranks: list[int]) -> PolicySamples:
    paths = resolve_rank_traces(trace_root, ranks)
    samples = {
        arm: {rank: read_boundaries(path) for rank, path in rank_paths.items()} for arm, rank_paths in paths.items()
    }
    reference_count = len(samples["baseline"][ranks[0]])
    for rank in ranks:
        baseline = samples["baseline"][rank]
        if len(baseline) != reference_count:
            raise ValueError(f"{paths['baseline'][rank]}: microbatch count differs across ranks")
        for arm in samples:
            candidate = samples[arm][rank]
            if candidate.keys() != baseline.keys():
                raise ValueError(f"{paths[arm][rank]}: microbatch boundaries differ from baseline")
            for key, sample in baseline.items():
                if candidate[key].tokens != sample.tokens:
                    index, phase = key
                    raise ValueError(f"{paths[arm][rank]}: input tokens differ at microbatch {index} {phase}")
    return samples


def format_mib(value: int, baseline: int | None = None) -> str:
    if baseline is None:
        return f"{value / MIB:.1f}"
    return f"{value / MIB:.1f} ({(value - baseline) / MIB:+.1f})"


def print_report(samples: PolicySamples, ranks: list[int]) -> None:
    arms = tuple(samples)
    baseline = samples["baseline"]
    microbatch_count = len(baseline[ranks[0]]) // len(PHASES)
    print("# FSDP2 CUDA allocator boundary samples")
    print()
    print("Values are PyTorch allocated MiB. Parentheses show the change from the matching baseline boundary.")
    print("Allocator samples are taken at host-side boundaries without synchronizing CUDA streams.")
    print("Use the unprofiled benchmark report for whole-step peak allocated memory and throughput.")
    print()
    print("| Rank | Microbatch | Phase | Input tokens | " + " | ".join(arms) + " |")
    print("|---:|---:|---|---:|" + "---:|" * len(arms))
    for rank in ranks:
        for key, reference in baseline[rank].items():
            index, phase = key
            phase_label = f"{phase} (final)" if phase == "after_backward" and index == microbatch_count else phase
            values = [
                format_mib(samples[arm][rank][key].allocated, None if arm == "baseline" else reference.allocated)
                for arm in arms
            ]
            print(f"| {rank} | {index} | {phase_label} | {reference.tokens} | " + " | ".join(values) + " |")

    print()
    print("Each cell shows the cumulative allocated peak through final backward / largest sampled reserved MiB.")
    print("The label gives the first sampled boundary by which the allocated peak had been reached.")
    print("The allocated peak excludes the subsequent optimizer step in this replay.")
    print()
    print("| Rank | " + " | ".join(arms) + " |")
    print("|---:|" + "---:|" * len(arms))
    for rank in ranks:
        values = []
        for by_rank in samples.values():
            boundaries = by_rank[rank]
            peak = boundaries[microbatch_count, "after_backward"].peak
            reserved = max(sample.reserved for sample in boundaries.values())
            index, phase = next(key for key, sample in boundaries.items() if sample.peak == peak)
            values.append(f"{format_mib(peak)} / {format_mib(reserved)} at {phase} #{index}")
        print(f"| {rank} | " + " | ".join(values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path, help="diagnostic replay directory containing all four policy traces")
    parser.add_argument("--ranks", nargs="+", type=int, default=[0, 1, 2, 3], help="DP ranks to compare")
    args = parser.parse_args()
    try:
        samples = read_matched_boundaries(args.trace_root, args.ranks)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print_report(samples, args.ranks)


if __name__ == "__main__":
    main()
