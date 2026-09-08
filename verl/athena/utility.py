"""Constraint filtering and multi-metric utility scoring for Athena probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

METRIC_DIRECTIONS = {
    "throughput": "maximize",
    "throughput_p10": "maximize",
    "mfu": "maximize",
    "gpu_utilization": "maximize",
    "throughput_per_gb": "maximize",
    "step_time": "minimize",
    "step_time_p95": "minimize",
    "gen_time": "minimize",
    "actor_time": "minimize",
    "peak_memory_gb": "minimize",
    "peak_reserved_memory_gb": "minimize",
    "device_memory_used_gb": "minimize",
    "device_memory_utilization": "minimize",
    "power_watts": "minimize",
    "energy_per_1k_tokens": "minimize",
    "ttft_ms": "minimize",
    "tpot_ms": "minimize",
    "kv_usage": "minimize",
    "transfer_queue_latency_ms": "minimize",
    "throughput_cv": "minimize",
    "duration_s": "minimize",
}

DEFAULT_WEIGHTS = {
    "throughput": 0.45,
    "throughput_p10": 0.15,
    "step_time": 0.15,
    "peak_memory_gb": 0.10,
    "throughput_cv": 0.15,
}


@dataclass(frozen=True)
class UtilitySummary:
    requested_weights: dict[str, float]
    effective_weights: dict[str, float]
    inactive_metrics: dict[str, str]
    constraints: dict[str, dict[str, float]]


class UtilityScorer:
    """Normalize available metrics to [0, 1] and compute weighted utility."""

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        constraints: Mapping[str, Mapping[str, float]] | None = None,
    ):
        self.weights = {str(key): float(value) for key, value in (weights or DEFAULT_WEIGHTS).items()}
        self.constraints = {
            str(metric): {str(operator): float(limit) for operator, limit in bounds.items()}
            for metric, bounds in (constraints or {}).items()
        }
        if not self.weights or all(weight == 0 for weight in self.weights.values()):
            raise ValueError("At least one utility weight must be positive")
        for metric, weight in self.weights.items():
            if metric not in METRIC_DIRECTIONS:
                raise ValueError(f"Unknown utility metric: {metric}")
            if weight < 0:
                raise ValueError(f"Utility weight must be non-negative: {metric}={weight}")
        for metric, bounds in self.constraints.items():
            for operator in bounds:
                if operator not in {"min", "max", "eq"}:
                    raise ValueError(f"Unknown constraint operator: {metric}.{operator}")

    def apply_constraints(self, results: Iterable[Any]) -> None:
        for result in results:
            violations = []
            if result.status != "success":
                result.constraint_violations = violations
                continue
            for metric, bounds in self.constraints.items():
                if not hasattr(result, metric):
                    raise ValueError(f"Unknown constraint metric: {metric}")
                raw_value = getattr(result, metric)
                if raw_value is None:
                    violations.append(f"{metric}=unavailable")
                    continue
                value = float(raw_value)
                if "min" in bounds and value < bounds["min"]:
                    violations.append(f"{metric}={value:.6g} < min={bounds['min']:.6g}")
                if "max" in bounds and value > bounds["max"]:
                    violations.append(f"{metric}={value:.6g} > max={bounds['max']:.6g}")
                if "eq" in bounds and value != bounds["eq"]:
                    violations.append(f"{metric}={value:.6g} != eq={bounds['eq']:.6g}")
            result.constraint_violations = violations

    def score(self, results: list[Any]) -> UtilitySummary:
        self.apply_constraints(results)
        feasible = [result for result in results if result.feasible]
        if not feasible:
            return UtilitySummary(self.weights, {}, {}, self.constraints)

        ranges: dict[str, tuple[float, float]] = {}
        inactive: dict[str, str] = {}
        for metric, weight in self.weights.items():
            if weight == 0:
                inactive[metric] = "weight is zero"
                continue
            raw_values = [getattr(result, metric) for result in feasible]
            missing_count = sum(value is None for value in raw_values)
            if missing_count:
                inactive[metric] = f"missing in {missing_count}/{len(feasible)} feasible trials"
                continue
            values = [float(value) for value in raw_values]
            low, high = min(values), max(values)
            if abs(high - low) <= 1e-12:
                inactive[metric] = f"no variation across feasible trials (value={low:.6g})"
                continue
            ranges[metric] = (low, high)

        active_weight = sum(self.weights[metric] for metric in ranges)
        if active_weight <= 0:
            # One feasible trial or identical observations: all are equivalent
            # under the requested utility rather than arbitrarily failing.
            for result in feasible:
                result.utility = 1.0
            return UtilitySummary(self.weights, {}, inactive, self.constraints)

        effective = {metric: self.weights[metric] / active_weight for metric in ranges}
        for result in feasible:
            components = {}
            utility = 0.0
            for metric, (low, high) in ranges.items():
                value = float(getattr(result, metric))
                normalized = (value - low) / (high - low)
                if METRIC_DIRECTIONS[metric] == "minimize":
                    normalized = 1.0 - normalized
                component = effective[metric] * normalized
                components[metric] = {
                    "raw": value,
                    "normalized": normalized,
                    "effective_weight": effective[metric],
                    "contribution": component,
                }
                utility += component
            result.utility = utility
            result.utility_components = components
        return UtilitySummary(self.weights, effective, inactive, self.constraints)
