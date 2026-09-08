from types import SimpleNamespace

import pytest

from verl.athena.offline_auto_tuner import parse_constraint_arguments, parse_weight_arguments
from verl.athena.utility import UtilityScorer


def result(**kwargs):
    defaults = {
        "status": "success",
        "constraint_violations": None,
        "throughput": 10.0,
        "step_time": 2.0,
        "peak_memory_gb": 4.0,
        "mfu": 0.0,
        "gpu_count": 2,
        "utility": None,
        "utility_components": None,
    }
    defaults.update(kwargs)
    item = SimpleNamespace(**defaults)
    item.feasible = item.status == "success"
    return item


def test_weighted_utility_handles_maximize_and_minimize_metrics():
    fast = result(throughput=20, step_time=1, peak_memory_gb=6)
    small = result(throughput=10, step_time=2, peak_memory_gb=2)
    scorer = UtilityScorer(weights={"throughput": 0.75, "peak_memory_gb": 0.25})

    summary = scorer.score([fast, small])

    assert fast.utility == pytest.approx(0.75)
    assert small.utility == pytest.approx(0.25)
    assert summary.effective_weights == {"throughput": 0.75, "peak_memory_gb": 0.25}


def test_all_zero_mfu_is_inactive_and_remaining_weights_are_renormalized():
    first = result(throughput=20, mfu=0)
    second = result(throughput=10, mfu=0)
    summary = UtilityScorer(weights={"mfu": 0.9, "throughput": 0.1}).score([first, second])

    assert "mfu" in summary.inactive_metrics
    assert summary.effective_weights == {"throughput": 1.0}
    assert first.utility == 1.0
    assert second.utility == 0.0


def test_nullable_metric_is_inactive_instead_of_becoming_zero():
    first = result(throughput=20, mfu=None)
    second = result(throughput=10, mfu=None)
    summary = UtilityScorer(weights={"mfu": 0.9, "throughput": 0.1}).score([first, second])

    assert summary.inactive_metrics["mfu"] == "missing in 2/2 feasible trials"
    assert summary.effective_weights == {"throughput": 1.0}


def test_generic_constraints_support_memory_and_gpu_count():
    accepted = result(peak_memory_gb=7.5, gpu_count=2)
    rejected = result(peak_memory_gb=8.5, gpu_count=1)
    scorer = UtilityScorer(
        weights={"throughput": 1.0},
        constraints={"peak_memory_gb": {"max": 8}, "gpu_count": {"eq": 2}},
    )
    scorer.apply_constraints([accepted, rejected])

    assert accepted.constraint_violations == []
    assert len(rejected.constraint_violations) == 2


def test_cli_weight_and_constraint_parsing():
    assert parse_weight_arguments(["throughput=0.1", "mfu=0.9"]) == {"throughput": 0.1, "mfu": 0.9}
    assert parse_constraint_arguments(["peak_memory_gb<=8", "gpu_count==2"]) == {
        "peak_memory_gb": {"max": 8.0},
        "gpu_count": {"eq": 2.0},
    }
