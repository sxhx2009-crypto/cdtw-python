"""Continuous Dynamic Time Warping (CDTW) for one-dimensional curves.

This module implements a convergent boundary-sampling approximation of the
L1-parameter-space CDTW definition from Buchin, Nusser, and Wong (SoCG 2022).
The input samples are interpreted as vertices of a one-dimensional polygonal
curve and the curve is parameterized by arc length.

Inside every pair of curve segments (a parameter-space cell), the optimal
continuous path cost between two sampled boundary points is evaluated in
closed form.  Thus no numerical quadrature is used; approximation enters only
through the sampled positions at which a path may cross a cell boundary.

The implementation is a practical reference implementation.  It is not the
paper's exact piecewise-quadratic boundary-propagation algorithm, which runs
in O(n^5) for a pair of one-dimensional curves of complexity at most n.

The sample indices are not time coordinates: only the one-dimensional value
curve remains after arc-length parameterization.  In particular, two constant
curves both have zero parameter length and therefore have CDTW integral zero
even when their constant values differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
FloatArray = NDArray[np.float64]


_DEFAULT_MEMORY_LIMIT_MIB = 512.0
# In-cell costs evaluate height**2, so the usable input range is bounded by the
# square root of the float64 maximum rather than by finiteness alone.
_MAX_SAFE_MAGNITUDE = float(np.sqrt(np.finfo(np.float64).max))
# A cell transition temporarily holds several dense float/bool matrices.  This
# deliberately conservative factor supplements the exact persistent-array
# estimate and prevents a single large cell from exhausting process memory.
_WORKING_BYTES_PER_TRANSITION_PAIR = 128


@dataclass(frozen=True)
class CDTWResult:
    """Result returned by :func:`cdtw` and :func:`cdtw_adaptive`.

    Attributes
    ----------
    distance:
        The approximated CDTW integral.  If ``normalized=True`` was used, this
        is divided by the total parameter-space path length.
    parameter_path:
        An ``(k, 2)`` array of arc-length coordinates ``(x, y)`` for the
        reconstructed continuous monotone matching, or ``None`` when no path
        was requested.
    value_path:
        An ``(k, 2)`` array containing the matched curve values at the points
        in ``parameter_path``, or ``None``.
    grid_shape:
        Number of parameter coordinates used for the two curves.
    step:
        Nominal arc-length spacing of the regular grid.  Original curve
        vertices are inserted too, so some actual intervals can be shorter.
    grid_size:
        Requested number of regular intervals along the longer curve.
    normalized:
        Whether the reported distance was divided by the total path length.
    converged:
        For adaptive runs, whether the requested number of consecutive
        resolution changes met the given tolerance.  This is an empirical
        stabilization flag, not a certificate against the continuous optimum.
        It is ``None`` for a single-resolution run.
    estimated_error:
        Maximum absolute change over the most recent adaptive convergence
        window.  This is a convergence diagnostic, not a certified error bound.
    history:
        ``(grid_size, distance)`` pairs produced during refinement.
    """

    distance: float
    parameter_path: FloatArray | None
    value_path: FloatArray | None
    grid_shape: tuple[int, int]
    step: float
    grid_size: int
    normalized: bool
    converged: bool | None = None
    estimated_error: float | None = None
    history: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class _ArcLengthCurve:
    vertices: FloatArray
    breaks: FloatArray

    @property
    def length(self) -> float:
        return float(self.breaks[-1])

    def evaluate(self, parameters: FloatArray) -> FloatArray:
        if self.length == 0.0:
            return np.full_like(parameters, self.vertices[0], dtype=np.float64)
        return np.interp(parameters, self.breaks, self.vertices)


def _as_curve(values: ArrayLike, name: str) -> _ArcLengthCurve:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    # In-cell costs square the height difference, so finite inputs are not
    # enough: values past sqrt(float64 max) overflow inside the cost matrix and
    # would otherwise return inf behind a burst of RuntimeWarnings.
    largest = float(np.max(np.abs(array)))
    if largest > _MAX_SAFE_MAGNITUDE:
        raise ValueError(
            f"{name} has values up to {largest:.3e}; CDTW costs square the "
            f"height difference, so magnitudes above {_MAX_SAFE_MAGNITUDE:.3e} "
            "overflow.  Rescale the inputs before calling."
        )

    # Repeated consecutive vertices form zero-length segments.  Removing them
    # preserves the polygonal curve and makes the arc-length coordinates
    # strictly increasing whenever the total length is positive.
    keep = np.concatenate(([True], np.diff(array) != 0.0))
    vertices = np.ascontiguousarray(array[keep], dtype=np.float64)

    if vertices.size > 2:
        # Consecutive segments pointing the same way are one straight piece of
        # the same one-dimensional curve, so only the turning points carry
        # shape.  Dropping the rest leaves the curve identical while removing
        # parameter-space cells, and it makes the result independent of how a
        # straight run happened to be sampled: without this, inserting a
        # collinear vertex split a cell and could only raise the cost, because
        # a path may cross a cell boundary solely at a sampled coordinate.
        directions = np.sign(np.diff(vertices))
        turning = np.concatenate(
            ([True], directions[1:] != directions[:-1], [True])
        )
        vertices = np.ascontiguousarray(vertices[turning], dtype=np.float64)

    if vertices.size == 1:
        breaks = np.array([0.0], dtype=np.float64)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            segment_lengths = np.abs(np.diff(vertices))
            breaks = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        if not np.all(np.isfinite(breaks)):
            raise ValueError(f"{name} has a non-finite or overflowing arc length")

    return _ArcLengthCurve(vertices=vertices, breaks=breaks)


def _grid_merge_tolerance(scale: float) -> float:
    """Spacing below which two arc-length coordinates denote the same point."""

    return 32.0 * np.finfo(np.float64).eps * max(1.0, scale)


def _unique_sorted(
    values: FloatArray, scale: float, anchors: FloatArray | None = None
) -> FloatArray:
    """Sort coordinates and merge only round-off-level near duplicates.

    ``anchors`` holds curve-vertex coordinates.  The cell decomposition and the
    closed-form in-cell cost are defined relative to those vertices, so an
    anchor must survive merging exactly: it absorbs nearby non-anchors instead
    of being absorbed.  Without this, a segment shorter than the merge
    tolerance lost its grid coordinate and :func:`_break_indices` raised
    ``RuntimeError`` -- the tolerance there is measured against the local
    coordinate, while merging is measured against the whole curve length.
    """

    values = np.sort(np.asarray(values, dtype=np.float64))
    tolerance = _grid_merge_tolerance(scale)
    anchor_set = (
        frozenset()
        if anchors is None
        else frozenset(float(anchor) for anchor in np.asarray(anchors, dtype=np.float64))
    )

    merged: list[float] = []
    is_anchor: list[bool] = []
    for raw in values:
        value = float(raw)
        anchored = value in anchor_set
        if merged and value - merged[-1] <= tolerance:
            if value == merged[-1]:
                is_anchor[-1] = is_anchor[-1] or anchored
                continue
            if is_anchor[-1]:
                if not anchored:
                    continue
                # Two distinct vertices this close: keep both rather than
                # silently deleting one of them.
            elif anchored:
                merged[-1] = value
                is_anchor[-1] = True
                continue
            else:
                # Keeping the larger value avoids leaving an endpoint just
                # short of the actual curve length.
                merged[-1] = value
                continue
        merged.append(value)
        is_anchor.append(anchored)
    return np.asarray(merged, dtype=np.float64)


def _with_endpoint(grid: FloatArray, length: float) -> FloatArray:
    """Trim a shared coordinate set so it ends exactly at ``length``.

    Used when both curves share one coordinate set: values at or beyond this
    curve's own length are dropped and the exact endpoint is appended, which
    keeps the result strictly increasing even when the two lengths differ by a
    rounding step.
    """

    result = np.concatenate(
        (grid[grid < length], np.array([length], dtype=np.float64))
    )
    result[0] = 0.0
    return result


def _joint_parameter_grids(
    curve1: _ArcLengthCurve,
    curve2: _ArcLengthCurve,
    grid_size: int,
) -> tuple[FloatArray, FloatArray, float]:
    max_length = max(curve1.length, curve2.length)
    if max_length == 0.0:
        zero = np.array([0.0], dtype=np.float64)
        return zero, zero.copy(), 0.0

    step = max_length / grid_size
    regular = np.linspace(0.0, max_length, grid_size + 1, dtype=np.float64)

    def reflected(values: FloatArray, length: float) -> FloatArray:
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, max_length)
        inside = values[(values >= -tolerance) & (values <= length + tolerance)]
        return np.clip(length - inside, 0.0, length)

    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, max_length)

    def curve_grid(length: float, own_breaks: FloatArray) -> FloatArray:
        if length == 0.0:
            return np.array([0.0], dtype=np.float64)
        regular_inside = regular[regular < length - tolerance]
        result = _unique_sorted(
            np.concatenate(
                (
                    regular_inside,
                    reflected(regular, length),
                    own_breaks,
                    reflected(own_breaks, length),
                    np.array([length]),
                )
            ),
            max_length,
            anchors=own_breaks,
        )
        result[0] = 0.0
        result[-1] = length
        return result

    x = curve_grid(curve1.length, curve1.breaks)
    y = curve_grid(curve2.length, curve2.breaks)

    # When the parameter lengths agree, sharing both coordinate sets preserves
    # the exact x=y matching of one geometric curve under different resampling.
    if abs(curve1.length - curve2.length) <= tolerance:
        common = _unique_sorted(
            np.concatenate((x, y)),
            max_length,
            anchors=np.concatenate((curve1.breaks, curve2.breaks)),
        )
        x = _with_endpoint(common, curve1.length)
        y = _with_endpoint(common, curve2.length)

    return x, y, step


def _validate_memory_limit(memory_limit_mib: float | None) -> float | None:
    if memory_limit_mib is None:
        return None
    if isinstance(memory_limit_mib, bool) or not isinstance(
        memory_limit_mib, (int, float, np.integer, np.floating)
    ):
        raise TypeError("memory_limit_mib must be a positive finite number or None")
    value = float(memory_limit_mib)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("memory_limit_mib must be a positive finite number or None")
    return value


def _estimate_peak_memory_mib(
    curve1: _ArcLengthCurve,
    curve2: _ArcLengthCurve,
    x: FloatArray,
    y: FloatArray,
    *,
    return_path: bool,
) -> float:
    """Conservatively estimate persistent plus largest-cell working memory."""

    cells = int(x.size) * int(y.size)
    persistent_arrays = 1 + (4 if return_path else 0)
    persistent_bytes = cells * np.dtype(np.float64).itemsize * persistent_arrays

    # Axis evaluation and integration create a handful of one-dimensional
    # arrays.  They are small next to the DP matrix but included for completeness.
    axis_bytes = 8 * np.dtype(np.float64).itemsize * (int(x.size) + int(y.size))

    working_bytes = 0
    if curve1.breaks.size > 1 and curve2.breaks.size > 1:
        scale = max(curve1.length, curve2.length)
        x_vertices = _break_indices(x, curve1.breaks, scale)
        y_vertices = _break_indices(y, curve2.breaks, scale)
        max_x_intervals = int(np.max(np.diff(x_vertices)))
        max_y_intervals = int(np.max(np.diff(y_vertices)))
        boundary_points = max_x_intervals + max_y_intervals + 1
        transition_pairs = boundary_points * boundary_points
        working_bytes = transition_pairs * _WORKING_BYTES_PER_TRANSITION_PAIR

    return float(persistent_bytes + axis_bytes + working_bytes) / (1024.0**2)


def _enforce_memory_limit(
    curve1: _ArcLengthCurve,
    curve2: _ArcLengthCurve,
    x: FloatArray,
    y: FloatArray,
    *,
    return_path: bool,
    memory_limit_mib: float | None,
) -> None:
    if memory_limit_mib is None:
        return
    estimated = _estimate_peak_memory_mib(
        curve1, curve2, x, y, return_path=return_path
    )
    if estimated > memory_limit_mib:
        raise MemoryError(
            "estimated peak CDTW working memory "
            f"({estimated:.1f} MiB) exceeds memory_limit_mib="
            f"{memory_limit_mib:.1f}; reduce grid_size, use return_path=False, "
            "or raise/disable the limit only after checking available memory"
        )


def _integral_mean_abs_linear(start: FloatArray, end: FloatArray) -> FloatArray:
    """Return ``integral_0^1 |(1-t) start + t end| dt`` elementwise."""

    abs_start = np.abs(start)
    abs_end = np.abs(end)
    same_side = start * end >= 0.0

    trapezoid = 0.5 * (abs_start + abs_end)
    denominator = abs_start + abs_end
    crossing = np.divide(
        0.5 * (abs_start * abs_start + abs_end * abs_end),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0.0,
    )
    return np.where(same_side, trapezoid, crossing)


def _break_indices(
    grid: FloatArray, breaks: FloatArray, scale: float
) -> NDArray[np.int64]:
    """Locate curve-segment endpoints in a grid, allowing round-off merging.

    ``scale`` is the longer curve's arc length.  The tolerance must be measured
    against that same scale as the grid merge tolerance, not against the local
    coordinate: a vertex near the start of a long curve has a small coordinate
    but is still stored with only ``eps * scale`` absolute resolution.
    """

    tolerance = 4.0 * _grid_merge_tolerance(scale)
    result: list[int] = []
    for value in breaks:
        right = int(np.searchsorted(grid, value))
        candidates = [index for index in (right - 1, right) if 0 <= index < grid.size]
        index = min(candidates, key=lambda candidate: abs(float(grid[candidate] - value)))
        if abs(float(grid[index] - value)) > tolerance:  # pragma: no cover
            raise RuntimeError("a curve vertex is missing from the parameter grid")
        result.append(index)
    return np.asarray(result, dtype=np.int64)


def _cell_cost_matrix(
    start_x: FloatArray,
    start_y: FloatArray,
    end_x: FloatArray,
    end_y: FloatArray,
    *,
    x0: float,
    y0: float,
    p0: float,
    q0: float,
    p_direction: float,
    q_direction: float,
) -> FloatArray:
    """Exact optimal in-cell costs from every start to every end point.

    Rows correspond to end points and columns to start points.  In an
    opposite-direction cell the height depends only on ``x+y``, hence every
    monotone path has the same cost.  In a same-direction cell, let signed
    height be ``r=P(x)-Q(y)``.  Along an L1-unit-speed path ``|r'| <= 1``.  For
    duration ``T``, the minimum integral is

        (|r0|^2 + |r1|^2)/2 - max(|r0|+|r1|-T, 0)^2/4.
    """

    sx = start_x[None, :]
    sy = start_y[None, :]
    tx = end_x[:, None]
    ty = end_y[:, None]
    delta_x = tx - sx
    delta_y = ty - sy
    duration = delta_x + delta_y
    tolerance = 128.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(end_x)), float(np.max(end_y))
    )
    feasible = (delta_x >= -tolerance) & (delta_y >= -tolerance)
    duration = np.maximum(duration, 0.0)

    start_height = (
        p0
        + p_direction * (sx - x0)
        - q0
        - q_direction * (sy - y0)
    )
    end_height = (
        p0
        + p_direction * (tx - x0)
        - q0
        - q_direction * (ty - y0)
    )

    if p_direction == q_direction:
        start_abs = np.abs(start_height)
        end_abs = np.abs(end_height)
        unreached_height = np.maximum(start_abs + end_abs - duration, 0.0)
        costs = (
            0.5 * (start_abs * start_abs + end_abs * end_abs)
            - 0.25 * unreached_height * unreached_height
        )
        costs = np.maximum(costs, 0.0)
    else:
        costs = duration * _integral_mean_abs_linear(start_height, end_height)

    return np.where(feasible, costs, np.inf)


def _axis_initialization(
    curve1: _ArcLengthCurve,
    curve2: _ArcLengthCurve,
    x: FloatArray,
    y: FloatArray,
    accumulated: FloatArray,
) -> None:
    accumulated[0, 0] = 0.0
    if x.size > 1:
        p = curve1.evaluate(x)
        q0 = curve2.vertices[0]
        horizontal = np.diff(x) * _integral_mean_abs_linear(
            p[:-1] - q0, p[1:] - q0
        )
        accumulated[1:, 0] = np.cumsum(horizontal)
    if y.size > 1:
        q = curve2.evaluate(y)
        p0 = curve1.vertices[0]
        vertical = np.diff(y) * _integral_mean_abs_linear(
            p0 - q[:-1], p0 - q[1:]
        )
        accumulated[0, 1:] = np.cumsum(vertical)


def _solve_boundary_sampled(
    curve1: _ArcLengthCurve,
    curve2: _ArcLengthCurve,
    x: FloatArray,
    y: FloatArray,
    return_path: bool,
) -> tuple[
    float,
    tuple[
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
    ]
    | None,
]:
    """Dynamic program over sampled boundaries of the original cells."""

    accumulated = np.full((x.size, y.size), np.inf, dtype=np.float64)
    _axis_initialization(curve1, curve2, x, y, accumulated)

    predecessor_data = None
    if return_path:
        previous_i = np.full(accumulated.shape, -1, dtype=np.int64)
        previous_j = np.full(accumulated.shape, -1, dtype=np.int64)
        previous_cell_i = np.full(accumulated.shape, -1, dtype=np.int64)
        previous_cell_j = np.full(accumulated.shape, -1, dtype=np.int64)
        if x.size > 1:
            previous_i[1:, 0] = np.arange(x.size - 1, dtype=np.int64)
            previous_j[1:, 0] = 0
        if y.size > 1:
            previous_i[0, 1:] = 0
            previous_j[0, 1:] = np.arange(y.size - 1, dtype=np.int64)
        predecessor_data = (
            previous_i,
            previous_j,
            previous_cell_i,
            previous_cell_j,
        )

    # A point curve has no parameter-space cells.  Its only possible matching
    # is already covered by the axis initialization above.
    if curve1.breaks.size == 1 or curve2.breaks.size == 1:
        return float(accumulated[-1, -1]), predecessor_data

    scale = max(curve1.length, curve2.length)
    x_vertices = _break_indices(x, curve1.breaks, scale)
    y_vertices = _break_indices(y, curve2.breaks, scale)

    for cell_i in range(curve1.vertices.size - 1):
        ix0 = int(x_vertices[cell_i])
        ix1 = int(x_vertices[cell_i + 1])
        p0 = float(curve1.vertices[cell_i])
        p_direction = float(np.sign(curve1.vertices[cell_i + 1] - p0))

        for cell_j in range(curve2.vertices.size - 1):
            iy0 = int(y_vertices[cell_j])
            iy1 = int(y_vertices[cell_j + 1])
            q0 = float(curve2.vertices[cell_j])
            q_direction = float(np.sign(curve2.vertices[cell_j + 1] - q0))

            # Input: bottom plus left (without duplicating bottom-left).
            input_i = np.concatenate(
                (
                    np.arange(ix0, ix1 + 1, dtype=np.int64),
                    np.full(iy1 - iy0, ix0, dtype=np.int64),
                )
            )
            input_j = np.concatenate(
                (
                    np.full(ix1 - ix0 + 1, iy0, dtype=np.int64),
                    np.arange(iy0 + 1, iy1 + 1, dtype=np.int64),
                )
            )

            # Output: top plus right (without duplicating top-right).
            output_i = np.concatenate(
                (
                    np.arange(ix0, ix1 + 1, dtype=np.int64),
                    np.full(iy1 - iy0, ix1, dtype=np.int64),
                )
            )
            output_j = np.concatenate(
                (
                    np.full(ix1 - ix0 + 1, iy1, dtype=np.int64),
                    np.arange(iy0, iy1, dtype=np.int64),
                )
            )

            input_costs = accumulated[input_i, input_j]
            local_costs = _cell_cost_matrix(
                x[input_i],
                y[input_j],
                x[output_i],
                y[output_j],
                x0=float(curve1.breaks[cell_i]),
                y0=float(curve2.breaks[cell_j]),
                p0=p0,
                q0=q0,
                p_direction=p_direction,
                q_direction=q_direction,
            )
            candidates = local_costs + input_costs[None, :]
            best_input = np.argmin(candidates, axis=1)
            best_cost = candidates[np.arange(output_i.size), best_input]
            old_cost = accumulated[output_i, output_j]

            # Strict improvement avoids a zero-length self predecessor at the
            # top-left and bottom-right corners of a cell.
            improve = best_cost < old_cost
            if np.any(improve):
                improved_i = output_i[improve]
                improved_j = output_j[improve]
                accumulated[improved_i, improved_j] = best_cost[improve]
                if predecessor_data is not None:
                    chosen = best_input[improve]
                    previous_i, previous_j, previous_cell_i, previous_cell_j = (
                        predecessor_data
                    )
                    previous_i[improved_i, improved_j] = input_i[chosen]
                    previous_j[improved_i, improved_j] = input_j[chosen]
                    previous_cell_i[improved_i, improved_j] = cell_i
                    previous_cell_j[improved_i, improved_j] = cell_j

    return float(accumulated[-1, -1]), predecessor_data


def _append_distinct(
    points: list[tuple[float, float]], point: tuple[float, float]
) -> None:
    if not points or point != points[-1]:
        points.append(point)


def _optimal_cell_path(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    x0: float,
    y0: float,
    p0: float,
    q0: float,
    p_direction: float,
    q_direction: float,
) -> list[tuple[float, float]]:
    """Construct a path attaining the closed-form in-cell cost."""

    if start == end:
        return [start]
    if p_direction != q_direction:
        # In an opposite-direction cell every monotone path has equal cost.
        return [start, end]

    sx, sy = start
    tx, ty = end
    start_height = p0 + p_direction * (sx - x0) - q0 - q_direction * (sy - y0)
    end_height = p0 + p_direction * (tx - x0) - q0 - q_direction * (ty - y0)
    duration = (tx - sx) + (ty - sy)
    start_abs = abs(start_height)
    end_abs = abs(end_height)
    minimum_abs_height = max(0.5 * (start_abs + end_abs - duration), 0.0)
    toward_duration = max(start_abs - minimum_abs_height, 0.0)
    away_duration = max(end_abs - minimum_abs_height, 0.0)
    valley_duration = max(duration - toward_duration - away_duration, 0.0)

    current_x, current_y = sx, sy
    path: list[tuple[float, float]] = [start]
    tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, duration)

    if toward_duration > tolerance:
        desired_height_rate = -float(np.sign(start_height))
        if desired_height_rate == p_direction:
            current_x += toward_duration
        else:
            current_y += toward_duration
        _append_distinct(path, (current_x, current_y))

    if valley_duration > tolerance:
        current_x += 0.5 * valley_duration
        current_y += 0.5 * valley_duration
        _append_distinct(path, (current_x, current_y))

    if away_duration > tolerance:
        desired_height_rate = float(np.sign(end_height))
        if desired_height_rate == p_direction:
            current_x += away_duration
        else:
            current_y += away_duration
        _append_distinct(path, (current_x, current_y))

    # Replace accumulated floating arithmetic with the exact sampled endpoint.
    _append_distinct(path, end)
    return path


def _reconstruct_path(
    curve1: _ArcLengthCurve,
    curve2: _ArcLengthCurve,
    x: FloatArray,
    y: FloatArray,
    predecessor_data: tuple[
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
    ],
) -> FloatArray:
    previous_i, previous_j, previous_cell_i, previous_cell_j = predecessor_data
    i = x.size - 1
    j = y.size - 1
    transitions: list[tuple[int, int, int, int, int, int]] = []
    safety_limit = x.size * y.size + 1

    while i != 0 or j != 0:
        pi = int(previous_i[i, j])
        pj = int(previous_j[i, j])
        if pi < 0 or pj < 0:  # pragma: no cover
            raise RuntimeError("missing predecessor while reconstructing CDTW path")
        transitions.append(
            (pi, pj, i, j, int(previous_cell_i[i, j]), int(previous_cell_j[i, j]))
        )
        i, j = pi, pj
        if len(transitions) > safety_limit:  # pragma: no cover
            raise RuntimeError("cycle detected while reconstructing CDTW path")

    points: list[tuple[float, float]] = []
    for pi, pj, i, j, cell_i, cell_j in reversed(transitions):
        start = (float(x[pi]), float(y[pj]))
        end = (float(x[i]), float(y[j]))
        if cell_i < 0 or cell_j < 0:
            local_path = [start, end]
        else:
            p0 = float(curve1.vertices[cell_i])
            q0 = float(curve2.vertices[cell_j])
            p_direction = float(np.sign(curve1.vertices[cell_i + 1] - p0))
            q_direction = float(np.sign(curve2.vertices[cell_j + 1] - q0))
            local_path = _optimal_cell_path(
                start,
                end,
                x0=float(curve1.breaks[cell_i]),
                y0=float(curve2.breaks[cell_j]),
                p0=p0,
                q0=q0,
                p_direction=p_direction,
                q_direction=q_direction,
            )
        for point in local_path:
            _append_distinct(points, point)

    if not points:  # both curves have zero arc length
        points = [(0.0, 0.0)]
    return np.asarray(points, dtype=np.float64)


def cdtw(
    curve1: ArrayLike,
    curve2: ArrayLike,
    *,
    grid_size: int = 256,
    return_path: bool = False,
    normalized: bool = False,
    memory_limit_mib: float | None = _DEFAULT_MEMORY_LIMIT_MIB,
) -> CDTWResult:
    """Approximate the CDTW distance between two 1-D polygonal curves.

    Parameters
    ----------
    curve1, curve2:
        One-dimensional finite numeric sequences.  Each sequence is treated as
        the ordered vertices of a polygonal curve in R.
    grid_size:
        Number of regular arc-length intervals along the longer curve, not
        per segment.  A curve with many segments therefore gets few regular
        samples inside each one; original vertex coordinates from both curves
        are always inserted, so every cell corner remains available.
        Memory is quadratic in the resulting grid dimensions; runtime also
        depends on how many boundary samples fall in each original cell.
    return_path:
        Reconstruct the monotone matching when true.  This stores four extra
        int64 predecessor arrays beside the cost array, so the persistent
        dynamic-programming memory is about five times larger.
    normalized:
        Divide the integral by ``length(curve1) + length(curve2)``.  The raw
        paper definition is returned by default.
    memory_limit_mib:
        Conservative peak-memory limit in MiB.  The default is 512 MiB.
        Set to ``None`` only to disable the guard intentionally.

    Returns
    -------
    CDTWResult
        Approximate distance plus optional matching and diagnostic metadata.

    Notes
    -----
    A larger ``grid_size`` enlarges the admissible family of numerical paths
    and generally improves the approximation.  Use :func:`cdtw_adaptive` when
    an empirical convergence check is preferred.
    """

    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)):
        raise TypeError("grid_size must be an integer")
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    checked_memory_limit = _validate_memory_limit(memory_limit_mib)

    first = _as_curve(curve1, "curve1")
    second = _as_curve(curve2, "curve2")
    x, y, step = _joint_parameter_grids(first, second, int(grid_size))
    _enforce_memory_limit(
        first,
        second,
        x,
        y,
        return_path=return_path,
        memory_limit_mib=checked_memory_limit,
    )
    if x.size == 1 and y.size == 1:
        raw_distance = 0.0
        predecessor_data = None
    else:
        raw_distance, predecessor_data = _solve_boundary_sampled(
            first, second, x, y, return_path
        )

    total_length = first.length + second.length
    distance = (
        raw_distance / total_length
        if normalized and total_length > 0.0
        else raw_distance
    )

    parameter_path = None
    value_path = None
    if return_path:
        if predecessor_data is None:
            parameter_path = np.array([[0.0, 0.0]], dtype=np.float64)
        else:
            parameter_path = _reconstruct_path(
                first, second, x, y, predecessor_data
            )
        value_path = np.column_stack(
            (
                first.evaluate(parameter_path[:, 0]),
                second.evaluate(parameter_path[:, 1]),
            )
        )

    return CDTWResult(
        distance=float(distance),
        parameter_path=parameter_path,
        value_path=value_path,
        grid_shape=(int(x.size), int(y.size)),
        step=float(step),
        grid_size=int(grid_size),
        normalized=bool(normalized),
        converged=None,
        estimated_error=None,
        history=((int(grid_size), float(distance)),),
    )


def cdtw_distance(
    curve1: ArrayLike,
    curve2: ArrayLike,
    *,
    grid_size: int = 256,
    normalized: bool = False,
    memory_limit_mib: float | None = _DEFAULT_MEMORY_LIMIT_MIB,
) -> float:
    """Convenience wrapper returning only the single-resolution distance."""

    return cdtw(
        curve1,
        curve2,
        grid_size=grid_size,
        return_path=False,
        normalized=normalized,
        memory_limit_mib=memory_limit_mib,
    ).distance


def cdtw_adaptive(
    curve1: ArrayLike,
    curve2: ArrayLike,
    *,
    initial_grid_size: int = 32,
    max_grid_size: int = 1024,
    rtol: float = 1e-4,
    atol: float = 1e-8,
    convergence_checks: int = 3,
    return_path: bool = False,
    normalized: bool = False,
    memory_limit_mib: float | None = _DEFAULT_MEMORY_LIMIT_MIB,
) -> CDTWResult:
    """Refine the CDTW grid until estimates empirically stabilize.

    A single unchanged pair can be a finite-grid plateau followed by a later
    decrease.  Therefore the default stopping rule requires three consecutive
    changes to satisfy ``abs(new - old) <= atol + rtol * abs(new)``.  The
    maximum change in that recent window is reported as ``estimated_error``.
    Neither this flag nor the reported change is a mathematical error bound on
    the true continuous optimum.

    Two plateau guards apply, because a run of identical values is common and
    on its own proves nothing.  A resolution step that leaves ``grid_shape``
    unchanged cannot change the value and is not counted as evidence, and an
    all-identical window reports the last observed movement rather than zero.
    Even so, a plateau spanning the whole window can still precede a further
    decrease, so treat ``converged=True`` at a small ``grid_size`` as weak.
    Richardson extrapolation is deliberately not used: the observed order of
    convergence is 2 only for simple inputs and is erratic in general, and on
    a plateau it would report an error estimate of exactly zero.

    ``converged`` is ``None`` when only one resolution was evaluated, since
    there is nothing to compare against.
    """

    for name, value in (
        ("initial_grid_size", initial_grid_size),
        ("max_grid_size", max_grid_size),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if initial_grid_size < 1:
        raise ValueError("initial_grid_size must be at least 1")
    if max_grid_size < initial_grid_size:
        raise ValueError("max_grid_size must be >= initial_grid_size")
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be a finite non-negative number")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a finite non-negative number")
    if isinstance(convergence_checks, bool) or not isinstance(
        convergence_checks, (int, np.integer)
    ):
        raise TypeError("convergence_checks must be an integer")
    if convergence_checks < 1:
        raise ValueError("convergence_checks must be at least 1")
    for name, value in (("return_path", return_path), ("normalized", normalized)):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a bool")
    checked_memory_limit = _validate_memory_limit(memory_limit_mib)

    history: list[tuple[int, float]] = []
    recent_differences: list[float] = []
    recent_passes: list[bool] = []
    previous: float | None = None
    previous_shape: tuple[int, int] | None = None
    last_movement: float | None = None
    estimated_error: float | None = None
    converged: bool | None = None
    size = int(initial_grid_size)
    last: CDTWResult | None = None

    while True:
        # The path is reconstructed in the loop rather than by repeating the
        # final resolution afterwards, which used to run the most expensive
        # dynamic program twice.
        last = cdtw(
            curve1,
            curve2,
            grid_size=size,
            return_path=return_path,
            normalized=normalized,
            memory_limit_mib=checked_memory_limit,
        )
        history.append((size, last.distance))

        if previous is not None:
            difference = abs(last.distance - previous)
            threshold = atol + rtol * abs(last.distance)
            if last.grid_shape == previous_shape:
                # Doubling grid_size need not add a coordinate once the curve
                # vertices dominate the grid.  An unchanged grid gives an
                # identical value by construction, so it is not evidence.
                difference = last_movement if last_movement is not None else 0.0
            elif difference > 0.0:
                last_movement = difference
            recent_differences.append(difference)
            recent_passes.append(difference <= threshold)
            window = recent_differences[-int(convergence_checks) :]
            estimated_error = max(window)
            if last_movement is not None and all(
                value == 0.0 for value in window
            ):
                # A run of identical values is a finite-grid plateau, not proof
                # of convergence: refinement often resumes decreasing later.
                # Never claim less than the last observed movement.
                estimated_error = last_movement
            converged = (
                len(window) == convergence_checks
                and all(recent_passes[-int(convergence_checks) :])
                and estimated_error <= threshold
            )
            if converged:
                break
        if size >= max_grid_size:
            if previous is not None:
                converged = bool(converged)
            break

        previous = last.distance
        previous_shape = last.grid_shape
        size = min(size * 2, int(max_grid_size))

    assert last is not None

    return CDTWResult(
        distance=last.distance,
        parameter_path=last.parameter_path,
        value_path=last.value_path,
        grid_shape=last.grid_shape,
        step=last.step,
        grid_size=last.grid_size,
        normalized=last.normalized,
        converged=converged,
        estimated_error=estimated_error,
        history=tuple(history),
    )


__all__ = ["CDTWResult", "cdtw", "cdtw_adaptive", "cdtw_distance"]
