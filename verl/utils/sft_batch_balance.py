# Copyright 2026 The verl Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Token-capped microbatch construction and calibrated SFT costs."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


MICROBATCH_POLICIES = ("ffd", "sorted", "repack")
# At eight samples each rank has at most 4,140 set partitions before token-cap pruning.
MAX_REPACK_SAMPLES_PER_RANK = 8


def partition_under_token_cap(lengths: list[int], limit: int) -> list[list[int]]:
    """Pack sample indices by first-fit decreasing under a hard token limit."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("The token limit must be a positive integer")
    partitions: list[list[int]] = []
    remaining: list[int] = []
    for index, length in enumerate(lengths):
        if type(length) is not int or not 0 < length <= limit:
            raise ValueError(f"Sample {index} has length {length!r} outside [1, {limit}]")

    for index in sorted(range(len(lengths)), key=lambda index: (-lengths[index], index)):
        length = lengths[index]
        for slot, capacity in enumerate(remaining):
            if length <= capacity:
                partitions[slot].append(index)
                remaining[slot] -= length
                break
        else:
            partitions.append([index])
            remaining.append(limit - length)
    return partitions


def split_to_microbatch_count(
    partitions: list[list[int]], lengths: list[int], count: int
) -> list[list[int]]:
    """Increase the bin count using subsets of existing bins to preserve the cap."""
    if not len(partitions) <= count <= len(lengths):
        raise ValueError(
            f"Cannot obtain {count} nonempty microbatches from "
            f"{len(partitions)} batches and {len(lengths)} samples"
        )
    result = [list(group) for group in partitions]
    if any(not group for group in result) or sorted(index for group in result for index in group) != list(
        range(len(lengths))
    ):
        raise ValueError("Partitions must contain each sample exactly once")

    totals = [sum(lengths[index] for index in group) for group in result]
    while len(result) < count:
        slot = max(
            (slot for slot, group in enumerate(result) if len(group) > 1),
            key=lambda slot: totals[slot],
        )
        groups: tuple[list[int], list[int]] = ([], [])
        split_totals = [0, 0]
        for index in result[slot]:
            side = int(split_totals[0] > split_totals[1])
            groups[side].append(index)
            split_totals[side] += lengths[index]
        result[slot] = groups[0]
        result.append(groups[1])
        totals[slot] = split_totals[0]
        totals.append(split_totals[1])
    return result


def _sample_costs(lengths: list[int]) -> list[int]:
    # Match verl.utils.seqlen_balancing.calculate_workload without a tensor round trip.
    return [24576 * length + length * length for length in lengths]


def _profile(groups: list[list[int]], costs: list[int]) -> tuple[int, ...]:
    return tuple(sum(costs[index] for index in group) for group in groups)


def _score_profiles(profiles: list[tuple[int, ...]]) -> tuple[int, int]:
    maximum_sum = spread_sum = 0
    for round_costs in zip(*profiles, strict=True):
        maximum = max(round_costs)
        maximum_sum += maximum
        spread_sum += maximum - min(round_costs)
    return maximum_sum, spread_sum


def microbatch_plan_proxy(rank_lengths: list[list[int]], plan: list[list[list[int]]]) -> tuple[int, int]:
    """Return sums of per-round maximum cost and cross-rank cost spread."""
    profiles = [_profile(groups, _sample_costs(lengths)) for lengths, groups in zip(rank_lengths, plan, strict=True)]
    return _score_profiles(profiles)


def _rank_packing_candidates(
    lengths: list[int], costs: list[int], count: int, limit: int
) -> dict[tuple[int, ...], list[list[int]]]:
    """Keep one feasible partition for each distinct descending cost profile."""
    sample_order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    candidates = {}
    groups: list[list[int]] = []
    totals: list[int] = []
    loads: list[int] = []

    def visit(position: int) -> None:
        if len(sample_order) - position < count - len(groups):
            return
        if position == len(sample_order):
            order = sorted(range(count), key=lambda slot: -loads[slot])
            profile = tuple(loads[slot] for slot in order)
            if profile not in candidates:
                candidates[profile] = [groups[slot].copy() for slot in order]
            return

        index = sample_order[position]
        length, cost = lengths[index], costs[index]
        equivalent_bins = set()
        for slot, group in enumerate(groups):
            state = totals[slot], loads[slot]
            if state in equivalent_bins or totals[slot] + length > limit:
                continue
            equivalent_bins.add(state)
            group.append(index)
            totals[slot] += length
            loads[slot] += cost
            visit(position + 1)
            group.pop()
            totals[slot] -= length
            loads[slot] -= cost

        if len(groups) < count:
            groups.append([index])
            totals.append(length)
            loads.append(cost)
            visit(position + 1)
            groups.pop()
            totals.pop()
            loads.pop()

    visit(0)
    return candidates


def plan_rank_microbatches(rank_lengths: list[list[int]], limit: int, policy: str) -> list[list[list[int]]]:
    """Plan local sample indices jointly while fixing FFD's aligned bin count.

    Repacking above the enumeration limit uses the fixed-bin sorted policy.
    """
    if policy not in MICROBATCH_POLICIES:
        raise ValueError(f"Unknown SFT microbatch policy {policy!r}. Choose one of {MICROBATCH_POLICIES}")
    if not rank_lengths or any(not lengths for lengths in rank_lengths):
        raise ValueError("The microbatch planner requires nonempty samples on every rank")

    initial = [partition_under_token_cap(lengths, limit) for lengths in rank_lengths]
    count = max(map(len, initial))
    baseline = [
        split_to_microbatch_count(groups, lengths, count)
        for groups, lengths in zip(initial, rank_lengths, strict=True)
    ]
    if policy == "ffd" or count == 1:
        return baseline

    costs = [_sample_costs(lengths) for lengths in rank_lengths]
    sorted_plan = []
    for groups, rank_costs in zip(baseline, costs, strict=True):
        order = sorted(range(count), key=lambda slot: sum(rank_costs[index] for index in groups[slot]), reverse=True)
        sorted_plan.append([groups[slot] for slot in order])
    if policy == "sorted" or len(rank_lengths) == 1 or any(
        len(lengths) > MAX_REPACK_SAMPLES_PER_RANK for lengths in rank_lengths
    ):
        return sorted_plan

    profiles = [_profile(groups, rank_costs) for groups, rank_costs in zip(sorted_plan, costs, strict=True)]
    original_profiles = profiles.copy()
    candidates = [
        _rank_packing_candidates(lengths, rank_costs, count, limit)
        for lengths, rank_costs in zip(rank_lengths, costs, strict=True)
    ]
    for choices, profile, groups in zip(candidates, profiles, sorted_plan, strict=True):
        choices[profile] = groups

    # Coordinate descent keeps other ranks fixed while selecting one rank's packing.
    for _ in range(4):
        before = profiles.copy()
        for rank, choices in enumerate(candidates):
            others = profiles[:rank] + profiles[rank + 1 :]
            bounds = [(min(round_costs), max(round_costs)) for round_costs in zip(*others, strict=True)]

            def score(candidate: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
                maximum_sum = minimum_sum = 0
                for cost, (other_min, other_max) in zip(candidate, bounds, strict=True):
                    maximum_sum += max(cost, other_max)
                    minimum_sum += min(cost, other_min)
                return maximum_sum, maximum_sum - minimum_sum, candidate

            profiles[rank] = min(choices, key=score)
        if profiles == before:
            break

    if _score_profiles(profiles) >= _score_profiles(original_profiles):
        return sorted_plan
    return [choices[profile] for profile, choices in zip(profiles, candidates, strict=True)]


@dataclass(frozen=True)
class SFTBatchCostModel:
    """Predict microbatch time in microseconds from packed sequence lengths."""

    intercept_us: float
    token_us: float
    length_squared_us: float

    def predict_us(self, lengths: list[int]) -> float:
        return (
            self.intercept_us
            + self.token_us * sum(lengths)
            + self.length_squared_us * sum(length * length for length in lengths)
        )

    def sample_weights(self, lengths: list[int]) -> list[int]:
        """Return additive per-sample cost weights in nanoseconds for partitioning."""
        return [
            max(1, round(1000 * (self.token_us * length + self.length_squared_us * length * length)))
            for length in lengths
        ]


def load_cost_model(path: str | Path) -> SFTBatchCostModel:
    """Load version 1 time coefficients expressed in raw token units."""
    with Path(path).open(encoding="utf-8") as stream:
        content = json.load(stream)
    if not isinstance(content, dict) or type(content.get("version")) is not int or content["version"] != 1:
        raise ValueError(f"{path}: expected cost model version 1")
    coefficients = []
    for key in ("intercept_us", "token_us", "length_squared_us"):
        value = content.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{path}: {key} must be finite and non-negative")
        coefficients.append(float(value))
    if not any(coefficients[1:]):
        raise ValueError(f"{path}: at least one length coefficient must be positive")
    return SFTBatchCostModel(*coefficients)
