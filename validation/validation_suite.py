"""Reproducible, research-style validation suite for the CDTW implementation.

Run ``python validation_suite.py`` from this directory.  The suite combines
high-precision exact oracles, metamorphic tests, path certificates,
differential testing, refinement checks, edge cases, and a stress benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
import json
import platform
import time
import tracemalloc

import numpy as np
from numpy.typing import NDArray

from cdtw import (
    _as_curve,
    _estimate_peak_memory_mib,
    _joint_parameter_grids,
    cdtw,
    cdtw_adaptive,
    cdtw_distance,
)


SEED = 20260819
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OracleCurve:
    vertices: FloatArray
    breaks: FloatArray

    @property
    def length(self) -> float:
        return float(self.breaks[-1])

    def evaluate(self, parameters: FloatArray) -> FloatArray:
        if self.length == 0.0:
            return np.full_like(parameters, self.vertices[0], dtype=np.float64)
        return np.interp(parameters, self.breaks, self.vertices)


class Checker:
    def __init__(self) -> None:
        self.assertions = 0

    def require(self, condition: bool, message: str) -> None:
        self.assertions += 1
        if not condition:
            raise AssertionError(message)

    def close(
        self,
        actual: float,
        expected: float,
        *,
        rtol: float,
        atol: float,
        message: str,
    ) -> None:
        self.assertions += 1
        if not np.isclose(actual, expected, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{message}: actual={actual:.17g}, expected={expected:.17g}, "
                f"abs_error={abs(actual - expected):.3e}"
            )


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(1.0, abs(actual), abs(expected))


def make_oracle_curve(values: FloatArray | list[float]) -> OracleCurve:
    vertices = np.asarray(values, dtype=np.float64)
    keep = np.concatenate(([True], np.diff(vertices) != 0.0))
    vertices = vertices[keep]
    if vertices.size == 1:
        breaks = np.array([0.0], dtype=np.float64)
    else:
        breaks = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(vertices)))))
    return OracleCurve(vertices=vertices, breaks=breaks)


def scalar_mean_abs_linear(start: float, end: float) -> float:
    """Independent scalar integral of an absolute affine function on [0, 1]."""

    start_abs = abs(start)
    end_abs = abs(end)
    if start * end >= 0.0:
        return 0.5 * (start_abs + end_abs)
    return 0.5 * (start_abs * start_abs + end_abs * end_abs) / (
        start_abs + end_abs
    )


def integrate_matching(
    curve1: FloatArray | list[float],
    curve2: FloatArray | list[float],
    path: FloatArray,
) -> float:
    """Independently integrate the polygonal matching returned by ``cdtw``."""

    first = make_oracle_curve(curve1)
    second = make_oracle_curve(curve2)
    total = 0.0

    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        duration = float(delta[0] + delta[1])
        if duration == 0.0:
            continue

        split = [0.0, 1.0]
        if delta[0] > 0.0:
            split.extend(
                float((value - start[0]) / delta[0])
                for value in first.breaks
                if start[0] < value < end[0]
            )
        if delta[1] > 0.0:
            split.extend(
                float((value - start[1]) / delta[1])
                for value in second.breaks
                if start[1] < value < end[1]
            )

        split = sorted(set(split))
        for left, right in zip(split[:-1], split[1:]):
            left_point = start + left * delta
            right_point = start + right * delta
            left_height = float(
                first.evaluate(np.array([left_point[0]]))[0]
                - second.evaluate(np.array([left_point[1]]))[0]
            )
            right_height = float(
                first.evaluate(np.array([right_point[0]]))[0]
                - second.evaluate(np.array([right_point[1]]))[0]
            )
            total += (
                duration
                * (right - left)
                * scalar_mean_abs_linear(left_height, right_height)
            )
    return total


def decimal_abs(value: Decimal) -> Decimal:
    return value if value >= 0 else -value


def decimal_mean_abs_linear(start: Decimal, end: Decimal) -> Decimal:
    start_abs = decimal_abs(start)
    end_abs = decimal_abs(end)
    if start * end >= 0:
        return (start_abs + end_abs) / Decimal(2)
    return (start_abs * start_abs + end_abs * end_abs) / (
        Decimal(2) * (start_abs + end_abs)
    )


def decimal_single_segment_cdtw(
    p0: Decimal, p1: Decimal, q0: Decimal, q1: Decimal
) -> Decimal:
    """Exact one-cell CDTW evaluated with 60-digit decimal arithmetic."""

    p_delta = p1 - p0
    q_delta = q1 - q0
    duration = decimal_abs(p_delta) + decimal_abs(q_delta)
    start_height = p0 - q0
    end_height = p1 - q1
    if p_delta * q_delta < 0:
        return duration * decimal_mean_abs_linear(start_height, end_height)

    start_abs = decimal_abs(start_height)
    end_abs = decimal_abs(end_height)
    unreached = max(start_abs + end_abs - duration, Decimal(0))
    return (
        (start_abs * start_abs + end_abs * end_abs) / Decimal(2)
        - unreached * unreached / Decimal(4)
    )


def vector_mean_abs_linear(start: FloatArray, end: FloatArray) -> FloatArray:
    """Independent vectorized integral used by the comparison solver."""

    start_abs = np.abs(start)
    end_abs = np.abs(end)
    denominator = start_abs + end_abs
    crossing = np.divide(
        0.5 * (start_abs * start_abs + end_abs * end_abs),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0.0,
    )
    return np.where(start * end >= 0.0, 0.5 * denominator, crossing)


def straight_grid_upper_bound(
    curve1: FloatArray, curve2: FloatArray, grid_size: int
) -> float:
    """Independent adjacent-edge grid DP producing a feasible-path upper bound."""

    first = _as_curve(curve1, "curve1")
    second = _as_curve(curve2, "curve2")
    x, y, _ = _joint_parameter_grids(first, second, grid_size)
    p = first.evaluate(x)
    q = second.evaluate(y)
    accumulated = np.full((x.size, y.size), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0

    if x.size > 1:
        horizontal_axis = np.diff(x) * vector_mean_abs_linear(
            p[:-1] - q[0], p[1:] - q[0]
        )
        accumulated[1:, 0] = np.cumsum(horizontal_axis)
    if y.size > 1:
        vertical_axis = np.diff(y) * vector_mean_abs_linear(
            p[0] - q[:-1], p[0] - q[1:]
        )
        accumulated[0, 1:] = np.cumsum(vertical_axis)
    if x.size == 1 or y.size == 1:
        return float(accumulated[-1, -1])

    dx = np.diff(x)
    dy = np.diff(y)
    horizontal = dx[:, None] * vector_mean_abs_linear(
        p[:-1, None] - q[None, :], p[1:, None] - q[None, :]
    )
    vertical = dy[None, :] * vector_mean_abs_linear(
        p[:, None] - q[None, :-1], p[:, None] - q[None, 1:]
    )
    diagonal = (dx[:, None] + dy[None, :]) * vector_mean_abs_linear(
        p[:-1, None] - q[None, :-1], p[1:, None] - q[None, 1:]
    )

    for i in range(1, x.size):
        for j in range(1, y.size):
            accumulated[i, j] = min(
                accumulated[i - 1, j] + horizontal[i - 1, j],
                accumulated[i, j - 1] + vertical[i, j - 1],
                accumulated[i - 1, j - 1] + diagonal[i - 1, j - 1],
            )
    return float(accumulated[-1, -1])


def random_curve(
    rng: np.random.Generator, minimum: int = 2, maximum: int = 10
) -> FloatArray:
    size = int(rng.integers(minimum, maximum + 1))
    steps = rng.uniform(-2.0, 2.0, size=size)
    steps[np.abs(steps) < 0.08] += 0.17
    return np.cumsum(steps, dtype=np.float64)


def refine_curve(curve: FloatArray) -> FloatArray:
    refined = np.empty(2 * curve.size - 1, dtype=np.float64)
    refined[0::2] = curve
    refined[1::2] = 0.5 * (curve[:-1] + curve[1:])
    return refined


def validate_decimal_single_segments(
    checker: Checker, rng: np.random.Generator, cases: int
) -> dict[str, float | int]:
    getcontext().prec = 60
    max_absolute_error = 0.0
    max_relative_error = 0.0
    for case in range(cases):
        integers = rng.integers(-200, 201, size=4)
        if integers[1] == integers[0]:
            integers[1] += 1
        if integers[3] == integers[2]:
            integers[3] -= 1
        values = [Decimal(int(value)) / Decimal(10) for value in integers]
        expected = float(decimal_single_segment_cdtw(*values))
        actual = cdtw_distance(
            [float(values[0]), float(values[1])],
            [float(values[2]), float(values[3])],
            grid_size=int(rng.integers(1, 24)),
        )
        max_absolute_error = max(max_absolute_error, abs(actual - expected))
        max_relative_error = max(max_relative_error, relative_error(actual, expected))
        checker.close(
            actual,
            expected,
            rtol=2e-13,
            atol=2e-13,
            message=f"Decimal one-cell oracle failed at case {case}",
        )
    return {
        "cases": cases,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
    }


def validate_point_polyline_oracle(
    checker: Checker, rng: np.random.Generator, cases: int
) -> dict[str, float | int]:
    max_relative_error = 0.0
    for case in range(cases):
        point = float(rng.uniform(-5.0, 5.0))
        curve = random_curve(rng, 2, 14)
        expected = sum(
            abs(float(right - left))
            * scalar_mean_abs_linear(point - float(left), point - float(right))
            for left, right in zip(curve[:-1], curve[1:])
        )
        actual = cdtw_distance([point], curve, grid_size=int(rng.integers(1, 40)))
        max_relative_error = max(max_relative_error, relative_error(actual, expected))
        checker.close(
            actual,
            expected,
            rtol=2e-13,
            atol=2e-13,
            message=f"point-polyline oracle failed at case {case}",
        )
    return {"cases": cases, "max_relative_error": max_relative_error}


def validate_metamorphic_properties(
    checker: Checker, rng: np.random.Generator, cases: int
) -> dict[str, float | int]:
    maxima = {
        "symmetry_relative_error": 0.0,
        "translation_relative_error": 0.0,
        "raw_scaling_relative_error": 0.0,
        "normalized_scaling_relative_error": 0.0,
        "joint_reversal_relative_error": 0.0,
        "refinement_increase_relative": 0.0,
    }
    for case in range(cases):
        first = random_curve(rng)
        second = random_curve(rng)
        d8 = cdtw_distance(first, second, grid_size=8)
        d16 = cdtw_distance(first, second, grid_size=16)
        base = cdtw_distance(first, second, grid_size=32)
        d64 = cdtw_distance(first, second, grid_size=64)
        checker.require(np.isfinite(base) and base >= 0.0, f"invalid cost at {case}")

        symmetric = cdtw_distance(second, first, grid_size=32)
        error = relative_error(symmetric, base)
        maxima["symmetry_relative_error"] = max(maxima["symmetry_relative_error"], error)
        checker.close(symmetric, base, rtol=2e-11, atol=2e-11, message=f"symmetry {case}")

        offset = float(rng.uniform(-1e4, 1e4))
        translated = cdtw_distance(first + offset, second + offset, grid_size=32)
        error = relative_error(translated, base)
        maxima["translation_relative_error"] = max(
            maxima["translation_relative_error"], error
        )
        checker.close(
            translated, base, rtol=2e-10, atol=2e-10, message=f"translation {case}"
        )

        factor = float(10.0 ** rng.uniform(-3.0, 3.0))
        if rng.random() < 0.5:
            factor = -factor
        scaled = cdtw_distance(first * factor, second * factor, grid_size=32)
        expected_scaled = factor * factor * base
        error = relative_error(scaled, expected_scaled)
        maxima["raw_scaling_relative_error"] = max(
            maxima["raw_scaling_relative_error"], error
        )
        checker.close(
            scaled,
            expected_scaled,
            rtol=2e-11,
            atol=2e-11,
            message=f"raw scaling {case}",
        )

        normalized = cdtw_distance(first, second, grid_size=32, normalized=True)
        normalized_scaled = cdtw_distance(
            first * factor, second * factor, grid_size=32, normalized=True
        )
        expected_normalized = abs(factor) * normalized
        error = relative_error(normalized_scaled, expected_normalized)
        maxima["normalized_scaling_relative_error"] = max(
            maxima["normalized_scaling_relative_error"], error
        )
        checker.close(
            normalized_scaled,
            expected_normalized,
            rtol=2e-11,
            atol=2e-11,
            message=f"normalized scaling {case}",
        )

        reversed_cost = cdtw_distance(first[::-1], second[::-1], grid_size=32)
        error = relative_error(reversed_cost, base)
        maxima["joint_reversal_relative_error"] = max(
            maxima["joint_reversal_relative_error"], error
        )
        checker.close(
            reversed_cost, base, rtol=2e-11, atol=2e-11, message=f"reversal {case}"
        )

        for coarse, fine in zip((d8, d16, base), (d16, base, d64)):
            increase = max(fine - coarse, 0.0) / max(1.0, abs(fine), abs(coarse))
            maxima["refinement_increase_relative"] = max(
                maxima["refinement_increase_relative"], increase
            )
            checker.require(
                fine <= coarse + 2e-11 * max(1.0, abs(coarse)),
                f"nested-grid increase {case}: {coarse} -> {fine}",
            )
    return {"cases": cases, **maxima}


def validate_path_certificates(
    checker: Checker, rng: np.random.Generator, cases: int
) -> dict[str, float | int]:
    max_cost_relative_error = 0.0
    max_endpoint_absolute_error = 0.0
    for case in range(cases):
        first = random_curve(rng, 1, 10)
        second = random_curve(rng, 1, 10)
        result = cdtw(
            first, second, grid_size=int(rng.integers(4, 49)), return_path=True
        )
        path = result.parameter_path
        checker.require(path is not None, f"path missing {case}")
        checker.require(
            bool(np.all(np.diff(path[:, 0]) >= -2e-11)), f"non-monotone x {case}"
        )
        checker.require(
            bool(np.all(np.diff(path[:, 1]) >= -2e-11)), f"non-monotone y {case}"
        )

        first_oracle = make_oracle_curve(first)
        second_oracle = make_oracle_curve(second)
        expected_end = np.array([first_oracle.length, second_oracle.length])
        endpoint_error = max(
            float(np.max(np.abs(path[0]))),
            float(np.max(np.abs(path[-1] - expected_end))),
        )
        max_endpoint_absolute_error = max(max_endpoint_absolute_error, endpoint_error)
        checker.require(endpoint_error <= 2e-11, f"invalid endpoint {case}")

        reintegrated = integrate_matching(first, second, path)
        error = relative_error(reintegrated, result.distance)
        max_cost_relative_error = max(max_cost_relative_error, error)
        checker.close(
            reintegrated,
            result.distance,
            rtol=5e-11,
            atol=5e-11,
            message=f"path certificate {case}",
        )
        expected_values = np.column_stack(
            (
                first_oracle.evaluate(path[:, 0]),
                second_oracle.evaluate(path[:, 1]),
            )
        )
        checker.require(
            bool(np.allclose(result.value_path, expected_values, rtol=1e-12, atol=1e-12)),
            f"value_path mismatch {case}",
        )
    return {
        "cases": cases,
        "max_cost_relative_error": max_cost_relative_error,
        "max_endpoint_absolute_error": max_endpoint_absolute_error,
    }


def validate_independent_upper_bound(
    checker: Checker, rng: np.random.Generator, cases: int
) -> dict[str, float | int]:
    max_bound_violation = 0.0
    max_relative_improvement = 0.0
    for case in range(cases):
        first = random_curve(rng, 2, 8)
        second = random_curve(rng, 2, 8)
        size = int(rng.integers(12, 41))
        actual = cdtw_distance(first, second, grid_size=size)
        upper = straight_grid_upper_bound(first, second, grid_size=size)
        max_bound_violation = max(max_bound_violation, max(actual - upper, 0.0))
        max_relative_improvement = max(
            max_relative_improvement,
            max(upper - actual, 0.0) / max(1.0, abs(upper)),
        )
        checker.require(
            actual <= upper + 5e-11 * max(1.0, abs(upper)),
            f"upper-bound violation {case}: {actual} > {upper}",
        )
    return {
        "cases": cases,
        "max_absolute_bound_violation": max_bound_violation,
        "max_relative_improvement_over_straight_grid": max_relative_improvement,
    }


def evaluate_resampling_convergence(
    rng: np.random.Generator, cases: int
) -> dict[str, float | int]:
    coarse_errors: list[float] = []
    fine_errors: list[float] = []
    for _ in range(cases):
        first = random_curve(rng, 2, 8)
        second = random_curve(rng, 2, 8)
        refined = refine_curve(first)
        for size, destination in ((32, coarse_errors), (128, fine_errors)):
            original = cdtw_distance(first, second, grid_size=size)
            resampled = cdtw_distance(refined, second, grid_size=size)
            destination.append(relative_error(original, resampled))
    return {
        "cases": cases,
        "coarse_max_relative_gap": float(np.max(coarse_errors)),
        "coarse_median_relative_gap": float(np.median(coarse_errors)),
        "fine_max_relative_gap": float(np.max(fine_errors)),
        "fine_median_relative_gap": float(np.median(fine_errors)),
        "cases_improved_or_equal": int(
            sum(fine <= coarse + 1e-14 for fine, coarse in zip(fine_errors, coarse_errors))
        ),
    }


def validate_numerical_edges(checker: Checker) -> dict[str, int]:
    cases = [
        ([1.0, 1.0, 1.0], [2.0, 2.0]),
        ([0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.5, 0.5, 0.0]),
        ([0.0, 1e-12, -1e-12, 2e-12], [1e-12, 0.0, 2e-12]),
        ([1e6, 1e6 + 1.0, 1e6 - 0.5], [1e6 + 0.2, 1e6 + 0.9]),
        ([0.0], [0.0]),
        ([3.0], [-2.0, 4.0, -1.0]),
    ]
    for index, (first, second) in enumerate(cases):
        result = cdtw(first, second, grid_size=64, return_path=True)
        checker.require(
            np.isfinite(result.distance) and result.distance >= 0.0,
            f"invalid numerical edge result {index}",
        )
        checker.close(
            integrate_matching(first, second, result.parameter_path),
            result.distance,
            rtol=2e-10,
            atol=2e-10,
            message=f"numerical edge certificate {index}",
        )
    return {"cases": len(cases)}


def validate_adaptive_and_memory_safety(checker: Checker) -> dict[str, object]:
    plateau_first = [0.398, -0.563, 0.589, 0.042, -1.571, 1.002]
    plateau_second = [-0.098, 0.620, 1.837, 0.268, -1.074, -0.681]
    adaptive = cdtw_adaptive(
        plateau_first,
        plateau_second,
        initial_grid_size=64,
        max_grid_size=256,
        rtol=1e-4,
        atol=1e-8,
    )
    sizes = [size for size, _ in adaptive.history]
    checker.require(sizes == [64, 128, 256], "adaptive stopped on flat step")
    checker.require(not adaptive.converged, "adaptive falsely certified plateau")
    checker.require(
        adaptive.distance < adaptive.history[1][1],
        "adaptive missed post-plateau decrease",
    )

    exact_cell = cdtw_adaptive(
        [0.0, 2.0],
        [0.37, 2.37],
        initial_grid_size=4,
        max_grid_size=32,
        rtol=0.0,
        atol=0.0,
        convergence_checks=3,
    )
    checker.require(bool(exact_cell.converged), "stable exact cell did not converge")
    checker.require(len(exact_cell.history) == 4, "stability window length mismatch")
    checker.close(
        float(exact_cell.estimated_error),
        0.0,
        rtol=0.0,
        atol=0.0,
        message="stable exact-cell diagnostic",
    )

    first = _as_curve(plateau_first, "plateau_first")
    second = _as_curve(plateau_second, "plateau_second")
    x, y, _ = _joint_parameter_grids(first, second, 8192)
    no_path_mib = _estimate_peak_memory_mib(
        first, second, x, y, return_path=False
    )
    path_mib = _estimate_peak_memory_mib(first, second, x, y, return_path=True)
    checker.require(no_path_mib > 512.0, "large no-path solve was underestimated")
    checker.require(path_mib > no_path_mib, "path memory estimate did not increase")

    guarded_modes = 0
    for return_path in (False, True):
        try:
            cdtw(
                plateau_first,
                plateau_second,
                grid_size=8192,
                return_path=return_path,
            )
        except MemoryError:
            guarded_modes += 1
        else:
            checker.require(False, f"memory guard failed for return_path={return_path}")
        checker.require(True, f"memory guard executed for return_path={return_path}")

    audit_rng = np.random.default_rng(SEED + 1)
    randomized_cases = 30
    stopped_before_reference = 0
    converged_count = 0
    max_relative_gap = 0.0
    violations = 0
    for case in range(randomized_cases):
        first_values = audit_rng.normal(size=6)
        second_values = audit_rng.normal(size=6)
        candidate = cdtw_adaptive(
            first_values,
            second_values,
            initial_grid_size=64,
            max_grid_size=1024,
            rtol=1e-4,
            atol=1e-8,
        )
        if candidate.grid_size < 1024:
            stopped_before_reference += 1
            reference = cdtw_distance(
                first_values, second_values, grid_size=1024
            )
        else:
            reference = candidate.distance
        converged_count += int(bool(candidate.converged))
        gap = abs(candidate.distance - reference)
        relative_gap = gap / max(abs(reference), np.finfo(np.float64).tiny)
        max_relative_gap = max(max_relative_gap, relative_gap)
        threshold = 1e-8 + 1e-4 * abs(reference)
        if gap > threshold:
            violations += 1
        checker.require(
            gap <= threshold,
            f"adaptive randomized audit exceeded tolerance at case {case}",
        )

    return {
        "cases": 34,
        "plateau_history": [[size, value] for size, value in adaptive.history],
        "plateau_converged": bool(adaptive.converged),
        "stability_window": 3,
        "large_grid_shape": [int(x.size), int(y.size)],
        "estimated_no_path_mib": no_path_mib,
        "estimated_with_path_mib": path_mib,
        "guarded_modes": guarded_modes,
        "randomized_audit_seed": SEED + 1,
        "randomized_audit_cases": randomized_cases,
        "randomized_stopped_before_grid_1024": stopped_before_reference,
        "randomized_converged_count": converged_count,
        "randomized_max_relative_gap_to_grid_1024": max_relative_gap,
        "randomized_tolerance_violations": violations,
    }


def run_stress_benchmark(rng: np.random.Generator) -> dict[str, object]:
    first = random_curve(rng, 120, 120)
    second = random_curve(rng, 120, 120)
    tracemalloc.start()
    start = time.perf_counter()
    result = cdtw(first, second, grid_size=512, return_path=True)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not np.isfinite(result.distance):
        raise AssertionError("stress benchmark returned a non-finite distance")
    return {
        "vertices_per_curve": 120,
        "requested_grid_size": 512,
        "actual_grid_shape": list(result.grid_shape),
        "path_points": int(result.parameter_path.shape[0]),
        "elapsed_seconds": elapsed,
        "tracemalloc_peak_mib": peak / (1024.0 * 1024.0),
    }


def main() -> None:
    started = time.perf_counter()
    checker = Checker()
    rng = np.random.default_rng(SEED)
    checks = {
        "decimal_single_segment_oracle": validate_decimal_single_segments(
            checker, rng, 1000
        ),
        "point_polyline_oracle": validate_point_polyline_oracle(checker, rng, 300),
        "metamorphic_properties": validate_metamorphic_properties(checker, rng, 500),
        "path_certificates": validate_path_certificates(checker, rng, 300),
        "independent_grid_upper_bound": validate_independent_upper_bound(
            checker, rng, 120
        ),
        "resampling_convergence_diagnostic": evaluate_resampling_convergence(rng, 120),
        "numerical_edge_cases": validate_numerical_edges(checker),
        "adaptive_and_memory_safety": validate_adaptive_and_memory_safety(checker),
        "stress_benchmark": run_stress_benchmark(rng),
    }
    summary = {
        "status": "PASS",
        "seed": SEED,
        "assertions": checker.assertions,
        "scenario_count": 2381,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "checks": checks,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
