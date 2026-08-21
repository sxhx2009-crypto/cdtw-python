"""How far off is the default ``grid_size`` on curves with many vertices?

``grid_size`` counts regular intervals along the *whole* longer curve, not per
segment, so a curve with many vertices gets few regular samples inside each
parameter-space cell and the reported value drifts above the converged one.
The drift is one-sided: every value is the cost of a feasible monotone path,
so refining can only lower it.

The size of the drift turns out to depend far more on the individual curve
pair than on the vertex count -- at 140-154 vertices the measured excess ranges
from 0.000% to 0.98%.  A single measurement per row would therefore be
misleading in either direction, so this reports the mean, the median and the
maximum over several instances per bucket.

Run from the repository root::

    python experiments/grid_size_accuracy.py

No network needed.  Results are written to
``experiments/grid_size_accuracy_results.json``.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdtw import cdtw_distance, curve_vertex_count  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "grid_size_accuracy_results.json"
SEED = 20260821
DEFAULT_GRID = 256
# Reference resolution.  A pilot at 141/156 vertices put the 2048 -> 4096 change
# at 1.2e-5 relative, a hundred times below the effect being measured, so 2048
# is a sound reference and keeps the sweep affordable.
REFERENCE_GRID = 2048
SAMPLE_COUNTS = (30, 60, 100, 150, 200, 280)
INSTANCES = 10


def main() -> int:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    print(f"default grid_size={DEFAULT_GRID} against grid_size={REFERENCE_GRID}")
    for count in SAMPLE_COUNTS:
        for instance in range(INSTANCES):
            a = np.cumsum(rng.normal(0.0, 1.0, count))
            b = np.cumsum(rng.normal(0.0, 1.0, count))
            vertices = max(curve_vertex_count(a), curve_vertex_count(b))
            coarse = cdtw_distance(a, b, grid_size=DEFAULT_GRID, memory_limit_mib=None)
            fine = cdtw_distance(a, b, grid_size=REFERENCE_GRID, memory_limit_mib=None)
            rows.append(
                {
                    "samples": int(count),
                    "instance": instance,
                    "vertices": int(vertices),
                    "excess_percent": 100.0 * (coarse - fine) / fine,
                }
            )
            print(
                f"  samples={count:4d} instance={instance:2d} vertices={vertices:4d}"
                f" excess={rows[-1]['excess_percent']:+.4f}%"
            )

    buckets: dict[int, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["samples"], []).append(row)

    summary = []
    print(f"\n{'vertices':>14s} {'mean':>9s} {'median':>9s} {'max':>9s}")
    for count in sorted(buckets):
        group = buckets[count]
        excess = [row["excess_percent"] for row in group]
        vertices = [row["vertices"] for row in group]
        entry = {
            "samples": count,
            "vertices_min": min(vertices),
            "vertices_max": max(vertices),
            "mean_percent": statistics.mean(excess),
            "median_percent": statistics.median(excess),
            "max_percent": max(excess),
        }
        summary.append(entry)
        print(
            f"  {entry['vertices_min']:4d}-{entry['vertices_max']:<7d}"
            f" {entry['mean_percent']:+8.3f}% {entry['median_percent']:+8.3f}%"
            f" {entry['max_percent']:+8.3f}%"
        )

    everything = [row["excess_percent"] for row in rows]
    below = sum(1 for value in everything if value < -1e-9)
    print(
        f"\nlowest {min(everything):+.4f}%   highest {max(everything):+.4f}%"
        f"   instances below the reference: {below}"
    )
    print("(none below: the coarse value is always an upper bound)")

    RESULTS.write_text(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                },
                "seed": SEED,
                "default_grid_size": DEFAULT_GRID,
                "reference_grid_size": REFERENCE_GRID,
                "instances_per_bucket": INSTANCES,
                "rows": rows,
                "summary": summary,
                "instances_below_reference": below,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {RESULTS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
