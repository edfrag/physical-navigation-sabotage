from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def load_occupancy(path: str | Path, occupied_threshold: int = 127) -> np.ndarray:
    """Load an occupancy image as bool array, True=occupied."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    return arr <= occupied_threshold


def save_occupancy(path: str | Path, occupied: np.ndarray) -> None:
    arr = np.where(occupied, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)
