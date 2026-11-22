#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a paired four-GPU study of repacking at matched validation quality."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
from statistics import median
import time

from compare import summarize
from run_microbatch_study import (
    GIB,
    event,
    get_settings,
    interrupt_run,
    launcher_command,
    read_arm,
    read_json,
    run_training,
    task_spec,
    verify_plan,
    write_atomic,
    write_json,
)

CAL_STEPS, CAL_FREQ = 100, 20
EVAL_FREQ, VAL_SAMPLES = 50, 128
FROZEN_STEPS, FROZEN_DATASET_STEPS = 16, 336
MAIN_BLOCKS, DATASET_STEP_STRIDE = 4, 29
# Keep batch-32 training identities below the first reserved evaluation offset.
MAX_MAIN_STEPS = 65_536 // 32 - DATASET_STEP_STRIDE * (MAIN_BLOCKS - 1)
# Each value gives an engineering noise floor and the largest acceptable gate limit.
GATE_LIMITS = {"loss_abs": (0.005, 0.025), "grad_rel": (0.005, 0.02)}


def read_validation(path, steps, frequency):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    expected = sorted({*range(frequency, steps + 1, frequency), steps})
    if [row["step"] for row in rows] != expected:
        raise ValueError(f"{path}: expected validation points at {expected}")
    for row in rows:
        for field in ("loss", "wall_s"):
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid {field} at step {row['step']}")
        if row["wall_s"] == 0:
            raise ValueError(f"{path}: zero wall time at step {row['step']}")
    if any(a["wall_s"] >= b["wall_s"] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{path}: validation timestamps must be increasing")
    return rows


def paired_rows(label, first, second):
    verify_plan(label, first, second)
    for a, b in zip(first, second, strict=True):
        if a["rank_sample_lengths"] != b["rank_sample_lengths"]:
            raise ValueError(f"{label}: rank sample lengths differ at step {a['step']}")
    return list(zip(first, second, strict=True))


def numeric_gap(pairs):
    loss_abs = max(abs(a["loss"] - b["loss"]) for a, b in pairs)
    loss_rel = max(abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), 1e-12) for a, b in pairs)
    grad_abs = max(abs(a["grad_norm"] - b["grad_norm"]) for a, b in pairs)
    grad_rel = max(abs(a["grad_norm"] - b["grad_norm"]) / max(abs(a["grad_norm"]), 1e-12) for a, b in pairs)
    return {"loss_abs": loss_abs, "loss_rel": loss_rel, "grad_abs": grad_abs, "grad_rel": grad_rel}


def run_arm(output, settings, deadline, task, name, policy, lr, split=None, frequency=None):
    attempt = output / name
    attempt.mkdir(parents=True)
    command, env = launcher_command(settings, task, policy, attempt)
    command.extend((f"optim.lr={lr}", "optim.lr_scheduler_type=constant",
                    f"optim.lr_warmup_steps={20 if lr else 0}", "optim.lr_warmup_steps_ratio=0.0"))
    validation_path = attempt / "validation.jsonl"
    if split is not None:
        command.extend((f"data.val_files={split}", f"data.val_max_samples={VAL_SAMPLES}",
                        f"trainer.test_freq={frequency}", f'trainer.sft_validation_metrics_path="{validation_path}"'))
    event(output, "started", group=task["name"], arm=name, policy=policy, steps=task["steps"], lr=lr)
    started = time.monotonic()
    run_training(command, env, attempt, task, deadline, settings["memory_limit_gib"])
    elapsed = time.monotonic() - started
    rows = read_arm(attempt, task, policy, settings["memory_limit_gib"])
    validation = read_validation(validation_path, task["steps"], frequency) if split is not None else []
    stats = summarize(rows, task["warmup"])
    result = {
        "group": task["name"],
        "arm": name,
        "policy": policy,
        "steps": task["steps"],
        "dataset_steps": task["dataset_steps"],
        "supervised_tok_s": stats["loss_rate"],
        "p95_step_s": stats["p95"],
        "peak_gib": max(max(row["peak_bytes"]) for row in rows) / GIB,
        "wall_s": elapsed,
        "validation": validation,
    }
    write_json(attempt / "complete.json", result)
    event(output, "completed", group=task["name"], arm=name, policy=policy, wall_s=round(elapsed, 2))
    return rows, result


def crossing(result, target):
    return next((row for row in result["validation"] if row["loss"] <= target), None)


def render(state):
    lines = ["# Four-GPU repacking quality study", "", f"Status **{state['status']}**."]
    if error := state.get("error"):
        lines.append(error)
    if gate := state.get("gate"):
        lines.extend(("", "## Fixed-weight comparison", ""))
        for name in ("aa", "ab"):
            gap = gate[name]
            lines.append(f"- {name.upper()} max absolute loss gap {gap['loss_abs']:.6g}, "
                         f"max relative grad-norm gap {gap['grad_rel']:.4%}.")
        lines.append(f"- Gate cap after A/A calibration. Loss {gate['limits']['loss_abs']:.6g}, "
                     f"grad norm {gate['limits']['grad_rel']:.4%} relative.")
    if calibration := state.get("calibration"):
        first, last = calibration["validation"][0], calibration["validation"][-1]
        lines.extend(("", f"Calibration validation loss {first['loss']:.6f} → {last['loss']:.6f}."))
    if (target := state.get("target")) is not None:
        lines.extend(("", f"Fixed synthetic holdout target **{target:.6f}**. "
                      f"Validation every {EVAL_FREQ} updates, with {state['steps']} updates per arm.", "",
                      "| Block | Policy | Supervised tok/s | P95 step s | Peak GiB | Final val loss | "
                      "First target step | Train-loop wall s to target | Job wall s |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|"))
        for result in state["arms"]:
            hit = crossing(result, target)
            hit_step, hit_s = (hit["step"], f"{hit['wall_s']:.1f}") if hit else ("not reached", "—")
            lines.append(f"| {result['group']} | {result['policy']} | {result['supervised_tok_s']:.1f} | "
                         f"{result['p95_step_s']:.3f} | {result['peak_gib']:.2f} | "
                         f"{result['validation'][-1]['loss']:.6f} | {hit_step} | {hit_s} | {result['wall_s']:.1f} |")
        lines.extend(("", "Paired effects use completed blocks. Loss delta is repack minus FFD. "
                      "A positive wall-time savings favors repack when both arms hit target.", ""))
        times, losses, throughputs = [], [], []
        for block in range(1, MAIN_BLOCKS + 1):
            arms = {r["policy"]: r for r in state["arms"] if r["group"] == f"main-{block:02d}"}
            if arms.keys() != {"ffd", "repack"}:
                continue
            ffd, repack = arms["ffd"], arms["repack"]
            loss = repack["validation"][-1]["loss"] - ffd["validation"][-1]["loss"]
            throughput = repack["supervised_tok_s"] / ffd["supervised_tok_s"] - 1
            a, b = crossing(ffd, target), crossing(repack, target)
            time_text = "censored"
            if a and b:
                savings = 1 - b["wall_s"] / a["wall_s"]
                times.append(savings)
                time_text = f"{savings:+.2%}"
            losses.append(loss)
            throughputs.append(throughput)
            lines.append(f"- Block {block}. Setup-matched throughput {throughput:+.2%}, "
                         f"final loss {loss:+.6f}, time-to-target savings {time_text}.")
        if losses:
            lines.append(f"- Median across {len(losses)} completed blocks. "
                         f"Throughput {median(throughputs):+.2%}, final loss {median(losses):+.6f}.")
        if times:
            lines.append(f"- Median time-to-target savings **{median(times):+.2%}** in {len(times)} uncensored blocks.")
        lines.extend(("", "Time-to-target is sampled at validation calls and includes prior train-loop validation. "
                      "An arm hitting at the first call has no earlier observed crossing time."))
    return "\n".join(lines) + "\n"


def save(output, state):
    write_json(output / "results.json", state)
    write_atomic(output / "report.md", render(state))


def study(output, settings, lr):
    deadline = time.time() + settings["hours"] * 3600
    state = {"status": "running", "settings": settings, "lr": lr, "arms": []}
    save(output, state)
    try:
        frozen = task_spec("frozen", ("ffd", "repack"), FROZEN_STEPS, 0, FROZEN_DATASET_STEPS)
        ffd, _ = run_arm(output, settings, deadline, frozen, "frozen-ffd-a", "ffd", 0.0)
        aa, _ = run_arm(output, settings, deadline, frozen, "frozen-ffd-b", "ffd", 0.0)
        ab, _ = run_arm(output, settings, deadline, frozen, "frozen-repack", "repack", 0.0)
        aa_pairs = paired_rows("frozen FFD repeat", ffd, aa)
        ab_pairs = paired_rows("frozen repack", ffd, ab)
        if any(a["microbatch_plan"] != b["microbatch_plan"] for a, b in aa_pairs):
            raise ValueError("Frozen FFD repeats used different microbatch plans")
        if all(a["microbatch_plan"] == b["microbatch_plan"] for a, b in ab_pairs):
            raise ValueError("Frozen gate sampled no changed microbatch plans")
        aa_gap, ab_gap = numeric_gap(aa_pairs), numeric_gap(ab_pairs)
        limits = {key: min(cap, max(floor, 5 * aa_gap[key])) for key, (floor, cap) in GATE_LIMITS.items()}
        state["gate"] = {"aa": aa_gap, "ab": ab_gap, "limits": limits}
        if any(aa_gap[key] > cap or ab_gap[key] > limits[key] for key, (_, cap) in GATE_LIMITS.items()):
            raise ValueError("Fixed-weight numerical gate failed. See FFD-repeat and repack differences.")
        save(output, state)

        cal = task_spec("calibration", ("ffd",), CAL_STEPS, 10)
        rows, calibration = run_arm(output, settings, deadline, cal, "calibration", "ffd", lr,
                                    "mixed-calibration", CAL_FREQ)
        state["calibration"] = calibration
        first, last = calibration["validation"][0], calibration["validation"][-1]
        if last["loss"] >= first["loss"]:
            raise ValueError("Calibration validation loss did not decline at the selected learning rate")
        overhead_s = max(0.0, calibration["wall_s"] - sum(row["step_s"] for row in rows))
        steady_s = sum(row["step_s"] for row in rows[10:]) / len(rows[10:])
        per_arm_s = max(0, deadline - time.time()) * 0.98 / (2 * MAIN_BLOCKS)
        budgeted_steps = (per_arm_s - overhead_s) / (steady_s * 1.07)
        steps = math.floor(min(MAX_MAIN_STEPS, budgeted_steps) / EVAL_FREQ) * EVAL_FREQ
        if steps < 2 * EVAL_FREQ:
            raise ValueError("Remaining budget cannot support two validation points per paired arm")
        state["target"] = (first["loss"] + last["loss"]) / 2
        state["steps"] = steps
        save(output, state)
        event(output, "plan_frozen", main_steps=steps, target=state["target"], lr=lr)

        for block in range(1, MAIN_BLOCKS + 1):
            order = ("ffd", "repack") if block % 2 else ("repack", "ffd")
            task = task_spec(f"main-{block:02d}", order, steps, 10, steps + DATASET_STEP_STRIDE * (block - 1))
            base = None
            for policy in order:
                rows, result = run_arm(output, settings, deadline, task, f"{task['name']}-{policy}", policy, lr,
                                       "mixed-holdout", EVAL_FREQ)
                if base is not None:
                    paired_rows(task["name"], base, rows)
                base = rows
                state["arms"].append(result)
                save(output, state)
        state["status"] = "completed"
        event(output, "study_ended", main_arms=len(state["arms"]))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, KeyboardInterrupt) as error:
        state.update(status="stopped", error=str(error))
        event(output, "study_stopped", error=str(error), main_arms=len(state["arms"]))
        raise
    finally:
        save(output, state)
        print(f"Report {output / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_or_output", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--memory-limit-gib", type=float, default=35)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.report_only:
            if args.output is not None:
                parser.error("--report-only accepts just OUTPUT")
            print(render(read_json(args.model_or_output / "results.json")), end="")
            return
        if args.output is None:
            parser.error("pass MODEL and OUTPUT")
        if not all(math.isfinite(value) and value > 0 for value in (args.hours, args.memory_limit_gib, args.lr)):
            parser.error("--hours, --memory-limit-gib and --lr require positive finite values")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        settings = get_settings(args.model_or_output.resolve(), args.hours, args.memory_limit_gib)
        with (output / ".quality-study.lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (output / "results.json").exists():
                raise ValueError("Use a new output directory to run another study")
            signal.signal(signal.SIGTERM, interrupt_run)
            study(output, settings, args.lr)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, KeyboardInterrupt) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
