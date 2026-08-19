import unittest

import numpy as np

from cdtw import (
    _as_curve,
    _integral_mean_abs_linear,
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
