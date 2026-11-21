#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a time-budgeted, paired four-GPU SFT microbatch packing study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from itertools import permutations
import json
import math
import os
from pathlib import Path
import re
import signal
from statistics import mean, median
import subprocess
import sys
import tempfile
import time

from compare import require_number, summarize
from compare_comm import read_comm_rows, verify_plan

SCRIPT_DIR = Path(__file__).resolve().parent
POLICIES = ("ffd", "sorted", "repack")
CAP = 16_384
GIB = 2**30
PILOT_STEPS = 12
PILOT_WARMUP = 3
MAIN_WARMUP = 10
HOLDOUT_WARMUP = 5
PLAN_VERSION = 1


class BudgetExpired(RuntimeError):
    """The allocated time was consumed before an arm finished."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path, contents):
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", encoding="utf-8", delete=False
    ) as temporary:
        name = Path(temporary.name)
        temporary.write(contents)
    try:
        name.replace(path)
    finally:
        name.unlink(missing_ok=True)


def write_json(path, object_):
    write_atomic(path, json.dumps(object_, indent=2, allow_nan=False) + "\n")


def event(output, status, **fields):
    entry = {"utc": utc_now(), "status": status, **fields}
    with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, allow_nan=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def task_spec(name, policies, steps, warmup, dataset_steps=None, workload="mixed", cap=CAP, profile_step=None):
    return {
        "name": name,
        "policies": list(policies),
        "steps": steps,
        "warmup": warmup,
        "dataset_steps": steps if dataset_steps is None else dataset_steps,
        "workload": workload,
        "cap": cap,
        "profile_step": profile_step,
    }


def completed_path(output, task, policy):
    return output / task["name"] / policy / "complete.json"


def validate_microbatches(row, task, policy):
    if row.get("microbatch_policy") != policy:
        raise ValueError(f"step {row['step']} did not use {policy}")
    require_number(row.get("microbatch_plan_ms"), "microbatch_plan_ms")
    count = row["microbatch_count"][0]
    rank_lengths = row.get("rank_sample_lengths")
    plan = row.get("microbatch_plan")
    if not isinstance(plan, list) or not isinstance(rank_lengths, list) or len(plan) != 4 or len(rank_lengths) != 4:
        raise ValueError(f"step {row['step']} needs four rank lengths and plans")

    profiles = []
    for rank, (lengths, bins) in enumerate(zip(rank_lengths, plan, strict=True)):
        if (
            not isinstance(lengths, list)
            or len(lengths) != len(row["rank_sample_indices"][rank])
            or any(type(value) is not int or not 1 <= value <= task["cap"] for value in lengths)
            or sum(lengths) != row["rank_input_tokens"][rank]
            or not isinstance(bins, list)
            or len(bins) != count
            or any(not isinstance(group, list) or not group for group in bins)
        ):
            raise ValueError(f"step {row['step']} has invalid microbatch shape on rank {rank}")
        flat = [index for group in bins for index in group]
        if any(type(index) is not int for index in flat) or sorted(flat) != list(range(len(lengths))):
            raise ValueError(f"step {row['step']} has duplicate or missing sample indices on rank {rank}")
        if any(sum(lengths[index] for index in group) > task["cap"] for group in bins):
            raise ValueError(f"step {row['step']} exceeds the token cap on rank {rank}")
        costs = [length * (24_576 + length) for length in lengths]
        profiles.append([sum(costs[index] for index in group) for group in bins])

    rounds = list(zip(*profiles, strict=True))
    expected = {
        "microbatch_proxy_max": sum(max(round_) for round_ in rounds),
        "microbatch_proxy_spread": sum(max(round_) - min(round_) for round_ in rounds),
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise ValueError(f"step {row['step']} reported inconsistent cost proxies")


def read_arm(attempt, task, policy, memory_limit_gib):
    path = attempt / "run" / f"{task['workload']}-combined" / "steps.jsonl"
    rows = read_comm_rows(path, task["steps"], defer=True, retain=True)
    for row in rows:
        validate_microbatches(row, task, policy)
        if max(row["peak_bytes"]) > memory_limit_gib * GIB:
            raise ValueError(f"{path} exceeds the allocated memory limit on step {row['step']}")
    return rows


def verify_pair(first_policy, first, second_policy, second):
    verify_plan(f"{first_policy}/{second_policy}", first, second)
    policies = {first_policy, second_policy}
    for a, b in zip(first, second, strict=True):
        if a["rank_sample_lengths"] != b["rank_sample_lengths"]:
            raise ValueError(f"paired step {a['step']} has different rank sample lengths")
        if policies == {"ffd", "sorted"}:
            for first_bins, second_bins in zip(a["microbatch_plan"], b["microbatch_plan"], strict=True):
                if sorted(map(tuple, first_bins)) != sorted(map(tuple, second_bins)):
                    raise ValueError(f"sorted policy changed bin membership or within-bin order at step {a['step']}")
        if policies == {"sorted", "repack"}:
            sorted_row, repacked_row = (a, b) if first_policy == "sorted" else (b, a)
            objectives = ("microbatch_proxy_max", "microbatch_proxy_spread")
            if tuple(repacked_row[key] for key in objectives) > tuple(sorted_row[key] for key in objectives):
                raise ValueError(f"repack worsened the sorted objective at step {a['step']}")

    # Later trajectories may drift from the different BF16 accumulation order.
    # Gate the same initial update and separately report whole-run loss / grad differences.
    for field, relative in (("loss", 0.005), ("grad_norm", 0.02)):
        if not math.isclose(first[0][field], second[0][field], rel_tol=relative, abs_tol=1e-4):
            raise ValueError(f"first update {field} diverged for {first_policy}/{second_policy}")


def complete_arms(output, task):
    return {
        policy: read_json(marker)
        for policy in task["policies"]
        if (marker := completed_path(output, task, policy)).is_file()
    }


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def check_new_peaks(path, cursor, limit_gib):
    if not path.is_file():
        return cursor
    with path.open("rb") as stream:
        stream.seek(cursor)
        while (line := stream.readline()).endswith(b"\n"):
            row = json.loads(line)
            peak = max(row["peak_bytes"]) / GIB
            if peak > limit_gib:
                raise ValueError(f"step {row['step']} allocated {peak:.2f} GiB, limit {limit_gib:g} GiB")
            cursor = stream.tell()
    return cursor


def launcher_command(settings, task, policy, attempt):
    env = os.environ.copy()
    for name in ("MIXED_DATA", "NARROW_DATA", "RAY_ADDRESS"):
        env.pop(name, None)
    env.update(
        ARM_ONLY=f"{task['workload']}-combined",
        PYTHON=sys.executable,
        CUDA_VISIBLE_DEVICES=settings["devices"],
        GLOBAL_BATCH="32",
        CONTEXT_LEN=str(CAP),
        TOKEN_CAP=str(task["cap"]),
        STEPS=str(task["steps"]),
        WARMUP_STEPS=str(task["warmup"]),
        DATASET_STEPS=str(task["dataset_steps"]),
        DATALOADER_WORKERS="0",
        REPEAT_REVERSED="0",
    )
    run = attempt / "run"
    command = [
        "bash", str(SCRIPT_DIR / "run_comm_4gpu.sh"), settings["model"], str(run),
        f"trainer.sft_microbatch_policy={policy}",
    ]
    if task["profile_step"] is not None:
        step = task["profile_step"]
        command.extend(
            [
                f"trainer.profile_interval=[{step},{step}]",
                "profiler.enable=true",
                "profiler.tool=torch",
                "profiler.ranks=[0,1,2,3]",
                "profiler.tool_config.torch.contents=[cpu,cuda]",
                "trainer.sft_comm_probe=true",
                f'profiler.save_path="{run / "trace"}"',
            ]
        )
    return command, env


def run_training(command, env, attempt, task, deadline, memory_limit_gib):
    steps_log = attempt / "run" / f"{task['workload']}-combined" / "steps.jsonl"
    with (attempt / "launcher.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        cursor = 0
        try:
            while True:
                cursor = check_new_peaks(steps_log, cursor, memory_limit_gib)
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise BudgetExpired("time budget reached during training")
                try:
                    returncode = process.wait(timeout=min(5, remaining))
                except subprocess.TimeoutExpired:
                    continue
                if returncode:
                    raise RuntimeError(f"trainer exited {returncode}. See {attempt / 'launcher.log'}")
                break
        except BaseException:
            stop_process_group(process)
            raise


def run_arm(output, settings, task, policy, deadline):
    marker = completed_path(output, task, policy)
    if marker.is_file():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=marker.parent))
    relative = attempt.relative_to(output).as_posix()
    event(output, "started", group=task["name"], policy=policy, attempt=relative)
    started = time.monotonic()
    try:
        command, env = launcher_command(settings, task, policy, attempt)
        run_training(command, env, attempt, task, deadline, settings["memory_limit_gib"])
        rows = read_arm(attempt, task, policy, settings["memory_limit_gib"])
        for peer, done in complete_arms(output, task).items():
            peer_rows = read_arm(output / done["attempt"], task, peer, settings["memory_limit_gib"])
            verify_pair(peer, peer_rows, policy, rows)
        if task["profile_step"] is not None:
            standard = task_spec("pilot", POLICIES, PILOT_STEPS, PILOT_WARMUP)
            original = complete_arms(output, standard)[policy]
            original_rows = read_arm(output / original["attempt"], standard, policy, settings["memory_limit_gib"])
            verify_plan(f"profile/{policy}", original_rows, rows)
            if any(a["microbatch_plan"] != b["microbatch_plan"] for a, b in zip(original_rows, rows, strict=True)):
                raise ValueError(f"profile/{policy} used a different microbatch plan from the pilot")
            traces = sorted((attempt / "run" / "trace").rglob("*prof_rank-*.json*"))
            ranks = [int(match[1]) for trace in traces if (match := re.search(r"prof_rank-(\d+)_", trace.name))]
            if sorted(ranks) != list(range(4)):
                raise ValueError(f"expected one trace per rank at {attempt / 'run' / 'trace'}")
        wall_s = time.monotonic() - started
        warmup = task["warmup"]
        measured = rows[warmup:]
        metrics = summarize(rows, warmup)
        metrics.update(
            planner_p50_ms=median(row["microbatch_plan_ms"] for row in measured),
            proxy_max=sum(row["microbatch_proxy_max"] for row in measured),
            proxy_spread=sum(row["microbatch_proxy_spread"] for row in measured),
            microbatches=sum(row["microbatch_count"][0] for row in measured),
            whole_arm_peak_gib=max(max(row["peak_bytes"]) for row in rows) / GIB,
        )
        result = {
            "attempt": relative,
            "wall_s": wall_s,
            "trainer_step_s": sum(row["step_s"] for row in rows),
            "measured_step_mean_s": mean(row["step_s"] for row in measured),
            "summary": metrics,
        }
        write_json(marker, result)
        event(output, "completed", group=task["name"], policy=policy, attempt=relative, wall_s=round(wall_s, 2))
    except (BudgetExpired, RuntimeError, OSError, ValueError, KeyError, TypeError, KeyboardInterrupt) as error:
        event(output, "attempt_failed", group=task["name"], policy=policy, attempt=relative, error=str(error))
        raise


def freeze_plan(output, settings, pilot):
    pilots = complete_arms(output, pilot).values()
    overhead = max(max(0, run["wall_s"] - run["trainer_step_s"]) for run in pilots)
    step_s = max(run["measured_step_mean_s"] for run in pilots)
    seconds = settings["hours"] * 3600
    launch_s = overhead * 1.15
    conservative_step_s = step_s * 1.07

    def step_budget(seconds_per_arm, warmup):
        steps = int((seconds_per_arm - launch_s) / conservative_step_s)
        return steps if steps > warmup else None

    primary = step_budget(seconds * 0.75 / 18, MAIN_WARMUP)
    holdout = step_budget(seconds * 0.125 / 6, HOLDOUT_WARMUP)
    groups = [pilot]
    if primary:
        for index, order in enumerate(permutations(POLICIES)):
            groups.append(
                task_spec(f"main-{index + 1:02d}", order, primary, MAIN_WARMUP, dataset_steps=primary + 29 * index)
            )
    if holdout:
        groups.extend(
            [
                task_spec("holdout-24k", ("repack", "sorted", "ffd"), holdout, HOLDOUT_WARMUP,
                          dataset_steps=holdout + 19, cap=24_576),
                task_spec("holdout-narrow", POLICIES, holdout, HOLDOUT_WARMUP,
                          dataset_steps=holdout + 37, workload="narrow"),
            ]
        )

    baseline = complete_arms(output, pilot)["ffd"]
    rows = read_arm(output / baseline["attempt"], pilot, "ffd", settings["memory_limit_gib"])[PILOT_WARMUP:]
    step = next((row["step"] for row in rows if row["microbatch_count"][0] == 3), None)
    if step is None:
        step = next((row["step"] for row in rows if row["microbatch_count"][0] > 1), None)
    if step is not None:
        groups.append(task_spec("profile", ("ffd", "repack"), PILOT_STEPS, PILOT_WARMUP, profile_step=step))
    plan = {
        "version": PLAN_VERSION,
        "pilot_step_s": step_s,
        "estimated_step_s": conservative_step_s,
        "estimated_launch_s": launch_s,
        "main_steps_per_arm": primary,
        "holdout_steps_per_arm": holdout,
        "groups": groups,
    }
    write_json(output / "plan.json", plan)
    return plan


def loss_gap(output, task, arms, settings):
    if any(policy not in arms for policy in POLICIES):
        return None
    rows = {
        policy: read_arm(output / arms[policy]["attempt"], task, policy, settings["memory_limit_gib"])
        for policy in POLICIES
    }
    return max(
        abs(base["loss"] - candidate["loss"])
        for policy in POLICIES[1:]
        for base, candidate in zip(rows["ffd"], rows[policy], strict=True)
    )


def build_report(output, settings, plan):
    groups = plan["groups"] if plan else [task_spec("pilot", POLICIES, PILOT_STEPS, PILOT_WARMUP)]
    lines = [
        "# Four-GPU microbatch packing study",
        "",
        f"Model `{settings['model']}`. GPU selection `{settings['devices']}`. Revision `{settings['revision']}`.",
        f"Budget **{settings['hours']:g} h**. Allocated memory limit **{settings['memory_limit_gib']:g} GiB**.",
        "All performance arms use a fixed communication policy and the synthetic dataset.",
        "An arm counts only after its telemetry and the available paired plans pass validation.",
        "",
    ]
    if plan:
        lines.extend(
            [
                f"Main steps per arm **{plan['main_steps_per_arm'] or 'not budgeted'}**. "
                f"Holdout steps per arm **{plan['holdout_steps_per_arm'] or 'not budgeted'}**.",
                "",
            ]
        )
    lines.extend([
        "| Group | Policy | Measured steps | Supervised tok/s | P50 step s | P95 step s | Peak allocated GiB "
        "| Planner P50 ms | Rank wall spread P50 ms | Microbatches | Proxy max | Proxy spread |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    paired = []
    missing = []
    traces = []
    for task in groups:
        name = task["name"]
        arms = complete_arms(output, task)
        absent = [policy for policy in task["policies"] if policy not in arms]
        if absent:
            missing.append(f"{name} ({', '.join(absent)})")
        if name == "profile":
            traces = [f"`{run['attempt']}/run/trace`" for run in arms.values()]
            continue
        for policy, run in arms.items():
            stats = run["summary"]
            lines.append(
                f"| {name} | {policy} | {stats['count']} | {stats['loss_rate']:.1f} | {stats['p50']:.3f} | "
                f"{stats['p95']:.3f} | {stats['whole_arm_peak_gib']:.2f} | {stats['planner_p50_ms']:.3f} | "
                f"{stats['wall_spread']:.3f} | {stats['microbatches']} | {stats['proxy_max']} | "
                f"{stats['proxy_spread']} |"
            )
        if absent:
            continue
        base = arms["ffd"]["summary"]
        deltas = [arms[policy]["summary"]["loss_rate"] / base["loss_rate"] - 1 for policy in POLICIES[1:]]
        repack = arms["repack"]["summary"]
        proxy = repack["proxy_max"] / base["proxy_max"] - 1
        paired.append((name, *deltas, proxy, loss_gap(output, task, arms, settings)))

    lines.extend(
        [
            "",
            "| Completed group | Sorted vs FFD throughput | Repack vs FFD throughput "
            "| Repack vs FFD proxy max | Maximum paired loss delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, sorted_delta, repack_delta, proxy, loss in paired:
        lines.append(f"| {name} | {sorted_delta:+.2%} | {repack_delta:+.2%} | {proxy:+.2%} | {loss:.4g} |")
    main = [result for result in paired if result[0].startswith("main-")]
    if main:
        for policy, index in (("sorted", 1), ("repack", 2)):
            values = [result[index] for result in main]
            lines.extend(
                ["", f"Main {policy} paired gain median **{median(values):+.2%}** "
                 f"across {len(values)}/6 complete blocks (range {min(values):+.2%} to {max(values):+.2%})."]
            )
    lines.extend(["", "Proxy columns are length-derived estimates. Training throughput includes planning overhead."])
    if missing:
        lines.extend(["", "Pending or incomplete groups", "", *(f"- {group}" for group in missing)])
    if traces:
        lines.extend(["", "Profiler replays are excluded from the throughput table.", "",
                      *(f"- {trace}" for trace in traces)])
    if (output / "trace_report.md").is_file():
        lines.extend(["", "[Four-rank trace readout](trace_report.md)"])
    return "\n".join(lines) + "\n"


def update_report(output, settings, plan):
    write_atomic(output / "report.md", build_report(output, settings, plan))


def run_group(output, settings, task, deadline, plan):
    for policy in task["policies"]:
        run_arm(output, settings, task, policy, deadline)
        update_report(output, settings, plan)


def run_profiles(output, profile):
    summary_script = SCRIPT_DIR / "summarize_microbatch_trace.py"
    if not summary_script.is_file():
        return
    ffd, repack = (complete_arms(output, profile)[policy]["attempt"] for policy in ("ffd", "repack"))
    args = [str(output / attempt / "run" / "trace") for attempt in (ffd, repack)]
    result = subprocess.run(
        [sys.executable, str(summary_script), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        write_atomic(output / "trace_summary_error.log", result.stderr)
    else:
        write_atomic(output / "trace_report.md", result.stdout)


def estimate_group(task, plan, finished):
    remaining = len(task["policies"]) - len(finished)
    if task["profile_step"] is not None:
        return remaining * (plan["estimated_launch_s"] + plan["estimated_step_s"] * task["steps"] + 60)
    return remaining * (plan["estimated_launch_s"] + plan["estimated_step_s"] * task["steps"])


def launch_study(output, settings):
    initial = settings["started"]
    seconds = settings["hours"] * 3600
    deadline = initial + seconds
    pilot = task_spec("pilot", POLICIES, PILOT_STEPS, PILOT_WARMUP)
    plan_path = output / "plan.json"
    plan = read_json(plan_path) if plan_path.is_file() else None
    try:
        if not plan:
            run_group(output, settings, pilot, initial + seconds * 0.05, None)
            plan = freeze_plan(output, settings, pilot)
            event(output, "plan_frozen", pilot_step_s=round(plan["pilot_step_s"], 4))
            update_report(output, settings, plan)
        for task in plan["groups"][1:]:
            name = task["name"]
            finished = complete_arms(output, task)
            phase_end = deadline
            if name.startswith("main-"):
                phase_end -= seconds * 0.20
            elif name.startswith("holdout-"):
                phase_end -= seconds * 0.075
            pending = [policy for policy in task["policies"] if policy not in finished]
            if pending and estimate_group(task, plan, finished) > phase_end - time.time():
                event(output, "budget_skipped", group=name, pending=pending)
                continue
            try:
                run_group(output, settings, task, phase_end, plan)
                if name == "profile":
                    run_profiles(output, task)
            except BudgetExpired as error:
                event(output, "phase_expired", group=name, error=str(error))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                if name != "profile":
                    raise
                event(output, "profile_failed", group=name, error=str(error))
        expected = sum(len(task["policies"]) for task in plan["groups"] if task["name"] != "profile")
        complete = sum(
            len(complete_arms(output, task)) for task in plan["groups"] if task["name"] != "profile"
        )
        event(output, "study_ended", completed_performance_arms=complete, planned_performance_arms=expected)
    finally:
        update_report(output, settings, plan)
        print(f"Report {output / 'report.md'}", flush=True)


def get_settings(model, hours, memory_limit):
    if not (model / "config.json").is_file():
        raise ValueError(f"expected a local model config in {model}")
    devices = [device.strip() for device in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")]
    if len(devices) != 4 or len(set(devices)) != 4 or not all(devices):
        raise ValueError("set CUDA_VISIBLE_DEVICES to four distinct GPU IDs")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR.parents[2], capture_output=True, text=True, check=False
    )
    return {
        "version": PLAN_VERSION,
        "model": str(model),
        "hours": hours,
        "memory_limit_gib": memory_limit,
        "devices": ",".join(devices),
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
    }


def interrupt_run(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_or_output", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--memory-limit-gib", type=float, default=35)
    parser.add_argument(
        "--report-only", action="store_true", help="print reports from OUTPUT without starting training"
    )
    args = parser.parse_args()
    try:
        if args.report_only:
            if args.output is not None:
                parser.error("--report-only accepts just OUTPUT")
            output = args.model_or_output.resolve()
            settings = read_json(output / "settings.json")
            plan_path = output / "plan.json"
            print(build_report(output, settings, read_json(plan_path) if plan_path.is_file() else None), end="")
            return
        if args.output is None:
            parser.error("pass MODEL and OUTPUT")
        if not all(math.isfinite(value) and value > 0 for value in (args.hours, args.memory_limit_gib)):
            parser.error("--hours and --memory-limit-gib must be positive finite numbers")
        output = args.output.resolve()
        settings = get_settings(args.model_or_output.resolve(), args.hours, args.memory_limit_gib)
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".study.lock").open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(1, f"{output} is already running\n")
            path = output / "settings.json"
            if path.is_file():
                previous = read_json(path)
                if any(previous[key] != value for key, value in settings.items()):
                    raise ValueError("resume requires the same model, Git revision, devices, hours and memory limit")
                settings = previous
            else:
                settings["started"] = time.time()
                write_json(path, settings)
            print(f"Study directory {output}", flush=True)
            signal.signal(signal.SIGTERM, interrupt_run)
            launch_study(output, settings)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, KeyboardInterrupt) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
