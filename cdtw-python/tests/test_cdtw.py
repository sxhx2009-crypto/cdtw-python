import unittest

import numpy as np

from cdtw import (
    _as_curve,
    _integral_mean_abs_linear,
    _joint_parameter_grids,
    cdtw,
    cdtw_adaptive,
    cdtw_distance,
)


def _integrate_returned_path(curve1, curve2, path) -> float:
    """Independent exact integration of a returned polygonal matching."""

    first = _as_curve(curve1, "curve1")
    second = _as_curve(curve2, "curve2")
    total = 0.0

    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        duration = float(delta.sum())
        if duration == 0.0:
            continue

        split_parameters = [0.0, 1.0]
        if delta[0] > 0.0:
            split_parameters.extend(
                float((value - start[0]) / delta[0])
                for value in first.breaks
                if start[0] < value < end[0]
            )
        if delta[1] > 0.0:
            split_parameters.extend(
                float((value - start[1]) / delta[1])
                for value in second.breaks
                if start[1] < value < end[1]
            )

        split_parameters = sorted(set(split_parameters))
        for left, right in zip(split_parameters[:-1], split_parameters[1:]):
            left_point = start + left * delta
            right_point = start + right * delta
            left_height = first.evaluate(np.array([left_point[0]]))[0] - second.evaluate(
                np.array([left_point[1]])
            )[0]
            right_height = first.evaluate(np.array([right_point[0]]))[0] - second.evaluate(
                np.array([right_point[1]])
            )[0]
            mean_height = _integral_mean_abs_linear(
                np.array(left_height), np.array(right_height)
            )
            total += duration * (right - left) * float(mean_height)
    return total


class CDTWTests(unittest.TestCase):
    def test_identical_curves_have_zero_cost(self) -> None:
        curve = [0.0, 1.0, -0.5, 2.0]
        result = cdtw(curve, curve, grid_size=32, return_path=True)
        self.assertAlmostEqual(result.distance, 0.0, places=12)
        self.assertIsNotNone(result.parameter_path)
        np.testing.assert_allclose(
            result.parameter_path[:, 0], result.parameter_path[:, 1], atol=1e-14
        )

    def test_resampling_invariance_for_same_polyline(self) -> None:
        coarse = [0.0, 2.0]
        resampled = [0.0, 0.3, 1.0, 1.8, 2.0]
        self.assertAlmostEqual(
            cdtw_distance(coarse, resampled, grid_size=7), 0.0, places=12
        )

    def test_collinear_resampling_deviates_only_at_finite_resolution(self) -> None:
        # Inserting collinear vertices leaves the polygonal curve unchanged but
        # splits parameter-space cells, and a path may only cross a cell
        # boundary at a sampled coordinate.  The extra constraint can only
        # raise the cost, and the gap has to vanish under refinement.  The
        # older "resampling invariance" test compared two spellings of one
        # curve against itself, where the answer is zero either way, so it
        # could not observe this at all.
        curve1 = [1.834, 0.717, 0.299, 0.282, -1.425]
        curve2 = [1.911, -1.869]
        midpoints = [
            value
            for index in range(len(curve1) - 1)
            for value in (curve1[index], 0.5 * (curve1[index] + curve1[index + 1]))
        ]
        refined = midpoints + [curve1[-1]]

        # The largest grid needs more than the default guard allows: a two-cell
        # pair puts every boundary sample into one cell transition matrix.
        previous = None
        for grid_size, bound in ((128, 1e-2), (512, 1e-3), (2048, 1e-5)):
            coarse = cdtw_distance(
                curve1, curve2, grid_size=grid_size, memory_limit_mib=None
            )
            split = cdtw_distance(
                refined, curve2, grid_size=grid_size, memory_limit_mib=None
            )
            self.assertGreater(split, coarse - 1e-12)
            self.assertLess(abs(split - coarse) / coarse, bound)
            if previous is not None:
                self.assertLessEqual(coarse, previous + 1e-12)
            previous = coarse

    def test_vertex_survives_a_segment_shorter_than_the_merge_tolerance(self) -> None:
        # A segment far below eps times the total arc length used to be merged
        # out of the parameter grid, after which the vertex had no coordinate
        # and _break_indices raised RuntimeError.
        for exponent in range(6, 20):
            with self.subTest(exponent=exponent):
                value = cdtw_distance(
                    [0.0, 10.0**-exponent, 1e6], [0.0, 1e6], grid_size=32
                )
                self.assertTrue(np.isfinite(value))
        self.assertTrue(
            np.isfinite(cdtw_distance([0.0, 1e-6, 1e12], [0.0, 1e12], grid_size=64))
        )

    def test_curves_far_below_unit_scale_keep_their_integral(self) -> None:
        # The whole curve is shorter than the merge tolerance measured against
        # max(1, length).  Every sample used to collapse onto one coordinate,
        # which silently reported 0.0 (or inf) instead of the true integral.
        for exponent in range(10, 18):
            with self.subTest(exponent=exponent):
                length = 10.0**-exponent
                distance = cdtw_distance([0.0, length], [-1.0, -1.0], grid_size=24)
                exact = length * (1.0 + (1.0 + length)) / 2.0
                self.assertAlmostEqual(distance / exact, 1.0, places=9)

    def test_shared_grid_keeps_vertices_when_lengths_differ_by_one_ulp(self) -> None:
        # Nearly equal arc lengths make both curves share one coordinate set.
        # Trimming that set per curve has to keep it strictly increasing and
        # must not drop a vertex of either curve.
        length = 2.0
        longer = float(np.nextafter(length, np.inf))
        for curve1, curve2 in (
            ([0.0, 1.0, length], [0.0, 0.4, longer]),
            ([0.0, longer], [0.0, length]),
            ([0.0, float(np.nextafter(length, -np.inf)), length], [0.0, length]),
        ):
            with self.subTest(curve1=curve1, curve2=curve2):
                first = _as_curve(curve1, "curve1")
                second = _as_curve(curve2, "curve2")
                x, y, _ = _joint_parameter_grids(first, second, 16)
                for grid, curve in ((x, first), (y, second)):
                    self.assertTrue(np.all(np.diff(grid) > 0.0))
                    self.assertEqual(grid[0], 0.0)
                    self.assertEqual(grid[-1], curve.length)
                    for vertex in curve.breaks:
                        self.assertTrue(np.any(grid == vertex))

    def test_opposite_directions_have_closed_form_cost(self) -> None:
        # P(x)=x and Q(y)=2-y on [0,2].  Therefore h=|x+y-2| and
        # every monotone L1 path has integral int_0^4 |z-2| dz = 4.
        self.assertAlmostEqual(
            cdtw_distance([0.0, 2.0], [2.0, 0.0], grid_size=3),
            4.0,
            places=12,
        )

    def test_parallel_offset_segments(self) -> None:
        # P(x)=x, Q(y)=y+0.37 on [0,2].  The optimal path reaches the
        # zero-height valley, follows it, and leaves it; total cost is 0.37^2.
        # In-cell optimization makes this exact even when the valley does not
        # coincide with a sampling-grid coordinate.
        self.assertAlmostEqual(
            cdtw_distance([0.0, 2.0], [0.37, 2.37], grid_size=7),
            0.37**2,
            places=12,
        )

    def test_degenerate_point_against_segment(self) -> None:
        # int_0^2 |3-y| dy = 4
        self.assertAlmostEqual(
            cdtw_distance([3.0], [0.0, 2.0], grid_size=5), 4.0, places=12
        )

    def test_symmetry(self) -> None:
        p = [0.0, 1.5, -0.2, 0.7]
        q = [0.2, 0.9, -0.5]
        forward = cdtw_distance(p, q, grid_size=48)
        backward = cdtw_distance(q, p, grid_size=48)
        self.assertAlmostEqual(forward, backward, places=12)

    def test_joint_reversal_invariance(self) -> None:
        p = [0.2, 1.7, -0.4, 0.9]
        q = [-0.3, 0.6, 1.2, -0.8, 0.1]
        forward = cdtw_distance(p, q, grid_size=31)
        reversed_together = cdtw_distance(p[::-1], q[::-1], grid_size=31)
        self.assertAlmostEqual(forward, reversed_together, places=12)

    def test_normalization(self) -> None:
        raw = cdtw([0.0, 2.0], [2.0, 0.0], grid_size=4)
        normalized = cdtw(
            [0.0, 2.0], [2.0, 0.0], grid_size=4, normalized=True
        )
        self.assertAlmostEqual(normalized.distance, raw.distance / 4.0, places=12)

    def test_adaptive_metadata(self) -> None:
        result = cdtw_adaptive(
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.0],
            initial_grid_size=4,
            max_grid_size=16,
            atol=0.0,
            rtol=0.0,
        )
        self.assertGreaterEqual(len(result.history), 2)
        self.assertLessEqual(result.grid_size, 16)
        self.assertIsNotNone(result.estimated_error)

    def test_adaptive_does_not_stop_on_a_single_flat_step(self) -> None:
        # This case has an exactly flat 64 -> 128 transition followed by a
        # genuine decrease at 256.  A one-step stopping rule returns too early.
        p = [0.398, -0.563, 0.589, 0.042, -1.571, 1.002]
        q = [-0.098, 0.620, 1.837, 0.268, -1.074, -0.681]
        result = cdtw_adaptive(
            p,
            q,
            initial_grid_size=64,
            max_grid_size=256,
            rtol=1e-4,
            atol=1e-8,
        )
        self.assertEqual([size for size, _ in result.history], [64, 128, 256])
        self.assertFalse(result.converged)
        self.assertGreater(result.estimated_error, 0.0)
        self.assertLess(result.distance, result.history[1][1])

    def test_adaptive_requires_the_full_stability_window(self) -> None:
        result = cdtw_adaptive(
            [0.0, 2.0],
            [0.37, 2.37],
            initial_grid_size=4,
            max_grid_size=32,
            rtol=0.0,
            atol=0.0,
            convergence_checks=3,
        )
        self.assertTrue(result.converged)
        self.assertEqual([size for size, _ in result.history], [4, 8, 16, 32])
        self.assertEqual(result.estimated_error, 0.0)

    def test_memory_guard_rejects_large_estimated_allocation(self) -> None:
        with self.assertRaisesRegex(MemoryError, "estimated peak CDTW working memory"):
            cdtw(
                [0.0, 1.0],
                [0.0, 1.0],
                grid_size=64,
                memory_limit_mib=0.01,
            )
        # Users can explicitly disable the guard after checking their system.
        result = cdtw(
            [0.0, 1.0],
            [0.0, 1.0],
            grid_size=4,
            memory_limit_mib=None,
        )
        self.assertAlmostEqual(result.distance, 0.0, places=12)

    def test_reconstructed_path_attains_reported_cost(self) -> None:
        p = [0.2, 1.4, -0.7, 0.8, 0.1]
        q = [-0.1, 0.9, -0.4, 1.0]
        result = cdtw(p, q, grid_size=37, return_path=True)
        self.assertIsNotNone(result.parameter_path)
        path = result.parameter_path
        self.assertTrue(np.all(np.diff(path[:, 0]) >= -1e-12))
        self.assertTrue(np.all(np.diff(path[:, 1]) >= -1e-12))
        self.assertAlmostEqual(
            _integrate_returned_path(p, q, path), result.distance, places=11
        )

    def test_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            cdtw_distance([], [1.0])
        with self.assertRaises(ValueError):
            cdtw_distance([0.0, np.nan], [1.0])
        with self.assertRaises(ValueError):
            cdtw_distance([[0.0], [1.0]], [1.0])
        with self.assertRaises(ValueError):
            cdtw_distance([0.0], [1.0], grid_size=0)
        with self.assertRaises(ValueError):
            cdtw_distance([0.0], [1.0], memory_limit_mib=0.0)
        with self.assertRaises(TypeError):
            cdtw_adaptive([0.0], [1.0], convergence_checks=True)
        with self.assertRaises(ValueError):
            cdtw_adaptive([0.0], [1.0], convergence_checks=0)


if __name__ == "__main__":
    unittest.main()
