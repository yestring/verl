# SPDX-License-Identifier: Apache-2.0
"""Compare a same-workload FSDP2 gradient-sync and reshard policy experiment."""

import argparse
from pathlib import Path
from statistics import mean

from compare import read_rows, summarize, verify_match

POLICIES = (
    ("baseline", False, False),
    ("defer-grad", True, False),
    ("retain-params", False, True),
    ("combined", True, True),
)
PLAN_FIELDS = ("rank_input_tokens", "rank_sample_indices", "microbatch_count")


def require_four_ints(row, name, path):
    values = row.get(name)
    if not isinstance(values, list) or len(values) != 4 or any(type(value) is not int or value < 1 for value in values):
        raise ValueError(f"{path}: step {row['step']} has invalid four-rank {name}")
    return values


def read_comm_rows(path, expected_steps, defer, retain):
    rows = read_rows(path, expected_steps)
    for row in rows:
        rank_tokens = require_four_ints(row, "rank_input_tokens", path)
        if sum(rank_tokens) != row["input_tokens"]:
            raise ValueError(f"{path}: step {row['step']} rank token counts do not cover the input")
        indices = row.get("rank_sample_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != 4
            or any(not isinstance(rank, list) or not rank for rank in indices)
        ):
            raise ValueError(f"{path}: step {row['step']} has no four-rank sample plan")
        flattened = [index for rank in indices for index in rank]
        if any(type(index) is not int for index in flattened) or sorted(flattened) != list(range(len(flattened))):
            raise ValueError(f"{path}: step {row['step']} sample plan repeats or omits indices")

        rounds = require_four_ints(row, "microbatch_count", path)
        if len(set(rounds)) != 1:
            raise ValueError(f"{path}: step {row['step']} has unequal microbatch counts across ranks")
        gradients = require_four_ints(row, "grad_sync_rounds", path)
        reshards = require_four_ints(row, "reshard_rounds", path)
        if gradients != ([1] * 4 if defer else rounds) or reshards != ([1] * 4 if retain else rounds):
            raise ValueError(f"{path}: step {row['step']} logical round counts disagree with the requested policy")
    return rows


def verify_plan(label, base, other):
    verify_match(label, base, other)
    for first, second in zip(base, other, strict=True):
        for field in PLAN_FIELDS:
            if first[field] != second[field]:
                raise ValueError(f"{label}: step {first['step']} differs in {field}")


def print_summary(name, rows, warmup):
    measured = rows[warmup:]
    summary = summarize(rows, warmup)
    rounds = [sum(row[key][0] for row in measured) for key in ("microbatch_count", "grad_sync_rounds", "reshard_rounds")]
    print(
        f"| {name} | {summary['count']} | {summary['input_rate']:.1f} | {summary['loss_rate']:.1f} | "
        f"{summary['p50']:.3f} | {summary['p95']:.3f} | {summary['peak_gib']:.2f} | "
        f"{rounds[0]} | {rounds[1]} | {rounds[2]} |"
    )
    return summary


def print_effects(workload, suffix, arms, summaries, warmup):
    base_name = f"{workload}-baseline{suffix}"
    base, base_summary = arms[base_name], summaries[base_name]
    candidates = POLICIES[1:] if workload == "mixed" else POLICIES[-1:]
    print(f"\n{workload}{suffix}, relative to its baseline:")
    for policy, _, _ in candidates:
        name = f"{workload}-{policy}{suffix}"
        candidate, summary = arms[name], summaries[name]
        change = summary["loss_rate"] / base_summary["loss_rate"] - 1
        peak_change = summary["peak_gib"] - base_summary["peak_gib"]
        first, second = base[0], candidate[0]
        print(
            f"- {policy}: supervised throughput {change:+.2%}, peak {peak_change:+.2f} GiB. "
            f"First update loss {first['loss']:.7g} → {second['loss']:.7g} "
            f"(Δ {second['loss'] - first['loss']:+.3g}), grad norm "
            f"{first['grad_norm']:.7g} → {second['grad_norm']:.7g} "
            f"(Δ {second['grad_norm'] - first['grad_norm']:+.3g})."
        )

    if workload == "narrow":
        one_round = sum(row["microbatch_count"][0] == 1 for row in base[warmup:])
        print(f"- One-microbatch control covers {one_round}/{len(base[warmup:])} measured steps.")
        return

    times = {
        policy: mean(row["step_s"] for row in arms[f"{workload}-{policy}{suffix}"][warmup:]) * 1000
        for policy, _, _ in POLICIES
    }
    baseline, grad, retain, combined = (times[policy] for policy, _, _ in POLICIES)
    print(
        "- Mean step savings (ms): defer gradients with normal/retained parameters "
        f"{baseline - grad:+.1f}/{retain - combined:+.1f}; retain parameters with every/last gradient sync "
        f"{baseline - retain:+.1f}/{grad - combined:+.1f}."
    )
    print(f"- Combined savings beyond the sum of individual savings: {grad + retain - baseline - combined:+.1f} ms.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--repeat-reversed", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.warmup_steps < 0 or (args.expected_steps is not None and args.expected_steps <= args.warmup_steps):
        parser.error("need at least one measured step after nonnegative warmup")

    groups = [("mixed", "", POLICIES), ("narrow", "", (POLICIES[0], POLICIES[-1]))]
    if args.repeat_reversed:
        groups.append(("mixed", "-r2", POLICIES))
    arms = {}
    try:
        for workload, suffix, policies in groups:
            for policy, defer, retain in policies:
                name = f"{workload}-{policy}{suffix}"
                path = args.output_root / name / "steps.jsonl"
                arms[name] = read_comm_rows(path, args.expected_steps, defer, retain)
                verify_plan(name, arms[f"{workload}-baseline"], arms[name])
                if len(arms[name]) <= args.warmup_steps:
                    raise ValueError(f"{path}: no measured steps after warmup")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        parser.exit(1, f"{error}\n")

    print("| Arm | Steps | Input token/s | Supervised token/s | Step P50 s | Step P95 s | Peak GiB | "
          "Microbatches | Gradient sync rounds | Post-backward reshard rounds |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    summaries = {name: print_summary(name, rows, args.warmup_steps) for name, rows in arms.items()}
    print("\nAll paired arms matched batch identities, token counts, and ordered rank sample plans at every step.")
    print("The round counters are logical policy totals on one rank across measured steps. "
          "Profiler traces provide native collective counts.")
    for workload, suffix, _ in groups:
        print_effects(workload, suffix, arms, summaries, args.warmup_steps)


if __name__ == "__main__":
    main()
