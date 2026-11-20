# Copyright 2026 The verl Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fit non-negative packed-microbatch time coefficients from an SFT profile."""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


def read_profile(path):
    steps = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                step = row["step"]
                ranks = row["microbatches"]
                if type(step) is not int or step < 0 or step in steps:
                    raise ValueError(f"Invalid or duplicate step {step!r}")
                if not isinstance(ranks, list) or not ranks:
                    raise ValueError("Expected a nonempty list of rank profiles")

                samples = []
                for rank, batches in enumerate(ranks):
                    if not isinstance(batches, list) or not batches:
                        raise ValueError(f"Rank {rank} has no microbatch timings")
                    for batch in batches:
                        tokens, squared = batch["tokens"], batch["sum_squared_lengths"]
                        elapsed_ms = batch["elapsed_ms"]
                        if type(tokens) is not int or tokens <= 0:
                            raise ValueError(f"Rank {rank} has an invalid token count")
                        if type(squared) is not int or not tokens <= squared <= tokens**2:
                            raise ValueError(f"Rank {rank} has an invalid sum of squared lengths")
                        if type(elapsed_ms) not in (int, float) or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                            raise ValueError(f"Rank {rank} has an invalid elapsed time")
                        samples.append((tokens, squared, 1000.0 * elapsed_ms, rank))
                steps[step] = samples
            except (TypeError, KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return steps


def fit_nonnegative(features, observed):
    # With three columns, enumerating active sets solves nonnegative least squares
    # without an additional dependency. Scaling avoids mixing units badly.
    scales = np.max(np.abs(features), axis=0)
    features = features / scales
    best_error = np.inf
    best = None
    for width in range(1, features.shape[1] + 1):
        for columns in itertools.combinations(range(features.shape[1]), width):
            selected = features[:, columns]
            coeffs, _, rank, _ = np.linalg.lstsq(selected, observed, rcond=None)
            if rank != width or np.any(coeffs < 0):
                continue
            error = np.mean((selected @ coeffs - observed) ** 2)
            if error >= best_error or np.isclose(error, best_error, rtol=1e-10, atol=1e-10):
                continue
            best = np.zeros(features.shape[1])
            best[list(columns)] = coeffs
            best_error = error
    if best is None:
        raise ValueError("Could not fit non-negative time coefficients")
    return best / scales, int(np.linalg.matrix_rank(features))


def validation_metrics(features, observed, coefficients):
    errors = np.abs(features @ coefficients - observed)
    relative = 100 * errors / observed
    return {
        "median_ape_pct": float(np.median(relative)),
        "p95_ape_pct": float(np.percentile(relative, 95)),
        "mae_us": float(np.mean(errors)),
    }


def fit_profile(steps, holdout):
    ordered = sorted(steps)
    validation_count = max(1, math.ceil(len(ordered) * holdout))
    train_steps, validation_steps = ordered[:-validation_count], ordered[-validation_count:]
    if len(train_steps) < 2:
        raise ValueError("Need at least two calibration steps and one held-out step")

    def pack_records(ids):
        return np.asarray([sample for step in ids for sample in steps[step]], dtype=np.float64)

    train_rows, validation_rows = pack_records(train_steps), pack_records(validation_steps)
    shapes = np.unique(train_rows[:, :2], axis=0)
    if len(shapes) < 2:
        raise ValueError("Calibration steps need at least two different microbatch shapes")
    train_x = np.column_stack((np.ones(len(train_rows)), train_rows[:, :2]))
    val_x = np.column_stack((np.ones(len(validation_rows)), validation_rows[:, :2]))
    train_y, val_y = train_rows[:, 2], validation_rows[:, 2]
    coefficients, rank = fit_nonnegative(train_x, train_y)
    if not np.any(coefficients[1:] > 0):
        raise ValueError("No positive length-dependent cost was fitted. Check profile timing and shape diversity.")
    return {
        "version": 1,
        "intercept_us": float(coefficients[0]),
        "token_us": float(coefficients[1]),
        "length_squared_us": float(coefficients[2]),
        "calibration": {
            "train_steps": train_steps,
            "validation_steps": validation_steps,
            "train_microbatches": len(train_rows),
            "unique_train_shapes": len(shapes),
            "feature_matrix_rank": rank,
            "validation_microbatches": len(validation_rows),
            "validation": validation_metrics(val_x, val_y, coefficients),
            "validation_by_rank": {
                str(int(rank_id)): validation_metrics(
                    val_x[validation_rows[:, 3] == rank_id],
                    val_y[validation_rows[:, 3] == rank_id],
                    coefficients,
                )
                for rank_id in sorted(set(validation_rows[:, 3]))
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="Initial profiled steps to discard before fitting and holdout"
    )
    parser.add_argument("--holdout", type=float, default=0.25, help="Fraction of steps held out chronologically")
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if not 0 < args.holdout < 1:
        parser.error("--holdout must be between zero and one")
    try:
        steps = read_profile(args.profile)
        warmup_steps = sorted(steps)[: args.warmup_steps]
        for step in warmup_steps:
            del steps[step]
        model = fit_profile(steps, args.holdout)
        model["calibration"]["warmup_steps"] = warmup_steps
        args.output.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Cost model: {args.output}")
    calibration = model["calibration"]
    if calibration["feature_matrix_rank"] < 3:
        print("Calibration features are collinear. Interpret coefficients on the profiled shapes only.")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
