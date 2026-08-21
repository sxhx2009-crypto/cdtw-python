"""Small runnable example for the CDTW module."""

# Runnable straight from a fresh clone: Python puts this script's own
# directory on sys.path, not the repository root, so add the root too.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np

from cdtw import cdtw_adaptive  # noqa: E402


p = np.array([0.0, 1.0, 0.2, 1.5, 1.0])
q = np.array([0.0, 0.8, 0.4, 1.4, 1.0])

result = cdtw_adaptive(
    p,
    q,
    initial_grid_size=32,
    max_grid_size=256,
    rtol=1e-4,
    return_path=True,
)

print(f"CDTW distance: {result.distance:.8f}")
print(f"grid shape: {result.grid_shape}")
print(f"converged: {result.converged}")
print(f"estimated change: {result.estimated_error}")
print("history:", result.history)
print("first matching points (arc-length coordinates):")
print(result.parameter_path[:5])

