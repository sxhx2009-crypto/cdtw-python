"""Small runnable example for the CDTW module."""

import numpy as np

from cdtw import cdtw_adaptive


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

