# Copyright 2026 The verl Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Token-capped microbatch construction and calibrated SFT costs."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


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
