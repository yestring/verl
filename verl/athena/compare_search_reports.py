"""Compare Athena search results at a shared measured GPU budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def compare_reports(
    reports: list[dict[str, Any]],
    budget_gpu_s: float | None = None,
    target: float | None = None,
) -> dict[str, Any]:
    """Compare the best confirmed candidate within the same cost limit."""
    if len(reports) < 2 or any(not report["trials"] for report in reports):
        raise ValueError("At least two nonempty search reports are required.")
    fixed_keys = (
        "config_sha256",
        "resume_from",
        "rollout_mode",
        "objective",
        "constraints",
        "warmup_steps",
        "probe_steps",
        "gpu_count",
    )
    reference = reports[0]
    for report in reports[1:]:
        for key in fixed_keys:
            if report[key] != reference[key]:
                raise ValueError(f"Reports must agree on {key}.")
        for key in ("token_budgets", "sweep_checkpointing", "kernel_backward"):
            if report["search_space"][key] != reference["search_space"][key]:
                raise ValueError(f"Reports must agree on search_space.{key}.")
    if reference["rollout_mode"] == "exact_step_cached_rollout":
        reference_counts = []
        for report in reports:
            for trial in report["trials"]:
                counts = trial["step_token_counts"]
                if counts is None or trial["status"] != "success":
                    continue
                overlap = min(len(reference_counts), len(counts))
                if counts[:overlap] != reference_counts[:overlap]:
                    raise ValueError("Cached per-step token counts differ between reports.")
                if len(counts) > len(reference_counts):
                    reference_counts = counts

    max_common = min(report["trials"][-1]["cumulative_gpu_s"] for report in reports)
    budget = max_common if budget_gpu_s is None else budget_gpu_s
    if not math.isfinite(budget) or not 0 < budget <= max_common:
        raise ValueError(f"GPU budget must be positive and at most {max_common:.3f} s.")
    if target is not None and (not math.isfinite(target) or target <= 0):
        raise ValueError("Target throughput must be positive and finite.")

    results = []
    for report in reports:
        best = None
        time_to_target = None
        eligible_trials = 0
        for trial in report["trials"]:
            spent = trial["cumulative_gpu_s"]
            if spent > budget:
                break
            eligible_trials += 1
            score = trial["incumbent_processed_tokens_per_gpu_s"]
            if score is not None:
                best = score
                if target is not None and score >= target and time_to_target is None:
                    time_to_target = spent
        results.append(
            {
                "sampler": report["sampler"],
                "seed": report["seed"],
                "probes_within_budget": eligible_trials,
                "best_processed_tokens_per_gpu_s": best,
                "gpu_s_to_target": time_to_target,
            }
        )
    return {"shared_budget_gpu_s": budget, "target": target, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--budget-gpu-s", type=float)
    parser.add_argument("--target", type=float, help="Processed tokens per GPU second")
    args = parser.parse_args()
    try:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
        result = compare_reports(reports, args.budget_gpu_s, args.target)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
