# SPDX-License-Identifier: Apache-2.0
"""Compare like-for-like Ray SFT runs from the batching benchmark."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


ARMS = ("mixed-baseline", "mixed-calibrated", "narrow-baseline", "narrow-calibrated")
IDENTITY_FIELDS = ("step", "batch_sha256", "input_tokens", "loss_tokens")
NUMERIC_FIELDS = ("step_s", "balance_ms", "loss", "grad_norm")
RANK_FIELDS = ("rank_wall_ms", "peak_bytes")


def require_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}: {value!r}")


def read_rows(path, expected_steps):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if not rows or (expected_steps is not None and len(rows) != expected_steps):
        raise ValueError(f"{path}: unexpected step count {len(rows)} (expected {expected_steps})")
    for index, row in enumerate(rows, start=1):
        if type(row.get("step")) is not int or row["step"] != index:
            raise ValueError(f"{path}: expected step {index}")
        if not isinstance(row.get("batch_sha256"), str) or not row["batch_sha256"]:
            raise ValueError(f"{path}: step {index} has no batch identity")
        for name in ("input_tokens", "loss_tokens"):
            if type(row.get(name)) is not int or row[name] <= 0:
                raise ValueError(f"{path}: step {index} has invalid {name}")
        if row["loss_tokens"] > row["input_tokens"]:
            raise ValueError(f"{path}: step {index} has more supervised tokens than inputs")
        for name in NUMERIC_FIELDS:
            require_number(row.get(name), f"{path}: step {index} {name}")
        if row["step_s"] == 0:
            raise ValueError(f"{path}: step {index} took zero time")
        for name in RANK_FIELDS:
            values = row.get(name)
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError(f"{path}: step {index} has no four-rank {name}")
            for value in values:
                require_number(value, f"{path}: step {index} {name}")
    return rows


def verify_match(workload, baseline, candidate):
    if len(baseline) != len(candidate):
        raise ValueError(f"{workload}: arms have different step counts")
    for first, second in zip(baseline, candidate):
        for field in IDENTITY_FIELDS:
            if first[field] != second[field]:
                raise ValueError(f"{workload}: step {first['step']} differs in {field}")


def summarize(rows, warmup):
    measured = rows[warmup:]
    if not measured:
        raise ValueError("no measured steps after warmup")
    times = sorted(row["step_s"] for row in measured)
    losses = sum(row["loss_tokens"] for row in measured)
    return {
        "count": len(times),
        "input_rate": sum(row["input_tokens"] for row in measured) / sum(times),
        "loss_rate": losses / sum(times),
        "p50": median(times),
        "p95": times[math.ceil(len(times) * 0.95) - 1],
        "balance_p50": median(row["balance_ms"] for row in measured),
        "wall_spread": median(max(row["rank_wall_ms"]) - min(row["rank_wall_ms"]) for row in measured),
        "peak_gib": max(max(row["peak_bytes"]) for row in measured) / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()
    if args.warmup_steps < 0 or (args.expected_steps is not None and args.expected_steps <= args.warmup_steps):
        parser.error("need at least one measured step after nonnegative warmup")

    try:
        arms = {
            arm: read_rows(args.output_root / arm / "steps.jsonl", args.expected_steps)
            for arm in ARMS
        }
        workloads = ("mixed", "narrow")
        for workload in workloads:
            verify_match(workload, arms[f"{workload}-baseline"], arms[f"{workload}-calibrated"])
        summaries = {arm: summarize(rows, args.warmup_steps) for arm, rows in arms.items()}
    except (OSError, ValueError, TypeError, AttributeError) as error:
        parser.exit(1, f"{error}\n")

    print("| Arm | Steps | Input token/s | Supervised token/s | Step P50 s | Step P95 s | "
          "Balance P50 ms | Rank wall spread P50 ms | Peak GiB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, s in summaries.items():
        print(
            f"| {arm} | {s['count']} | {s['input_rate']:.1f} | {s['loss_rate']:.1f} | "
            f"{s['p50']:.3f} | {s['p95']:.3f} | {s['balance_p50']:.3f} | "
            f"{s['wall_spread']:.3f} | {s['peak_gib']:.2f} |"
        )

    print("\nBoth pairs matched batch identities and token counts at every step.")
    for workload in workloads:
        baseline, candidate = f"{workload}-baseline", f"{workload}-calibrated"
        delta = summaries[candidate]["loss_rate"] / summaries[baseline]["loss_rate"] - 1
        first, second = arms[baseline][0], arms[candidate][0]
        print(
            f"{workload} supervised throughput change {delta:+.2%}. First update "
            f"loss {first['loss']:.6g} → {second['loss']:.6g}, "
            f"grad norm {first['grad_norm']:.6g} → {second['grad_norm']:.6g}."
        )


if __name__ == "__main__":
    main()
