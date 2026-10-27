from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def disk(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def inflate(occupied: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary hard-cutoff inflation used by the offline surrogate."""
    if radius_cells == 0:
        return occupied.copy()
    return binary_dilation(occupied, structure=disk(radius_cells))
