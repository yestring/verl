import json
from pathlib import Path

import pytest

from verl.athena.offline_auto_tuner import load_probe_states, summarize_probe


def state(step, throughput, step_time, peak_memory, utilization=0.5, mfu=0.0):
    return {
        "global_step": step,
        "throughput": throughput,
        "total_step_time": step_time,
        "gpu": {"memory_peak_gb": peak_memory, "utilization": utilization},
        "training": {"mfu": mfu},
    }


def test_structured_probe_metrics_discard_warmup_and_use_robust_aggregates(tmp_path: Path):
    path = tmp_path / "trace_step_1.jsonl"
    records = [
        state(1, 1, 99, 10),
        state(2, 100, 2.0, 20, 0.7, 0.4),
        state(3, 120, 1.5, 22, 0.9, 0.5),
        state(4, 110, 1.7, 21, 0.8, 0.45),
    ]
    path.write_text("\n".join(json.dumps(item) for item in records))

    summary = summarize_probe(load_probe_states(tmp_path), warmup_steps=1)
    assert summary["throughput"] == 110.0
    assert summary["step_time"] == 1.7
    assert summary["peak_memory_gb"] == 22.0
    assert summary["gpu_utilization"] == pytest.approx(0.8)
    assert summary["mfu"] == 0.45
    assert summary["samples"] == 3
