"""Load, manipulate, and save Nav2 occupancy grid maps."""

import numpy as np
from PIL import Image
import yaml
import copy
from pathlib import Path
from typing import Optional, Tuple


class OccupancyMap:
    """Load and manipulate Nav2 occupancy grid maps (.yaml + .pgm)."""

    FREE = 0
    OCCUPIED = 100
    UNKNOWN = -1

    def __init__(self, yaml_path: str):
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            self.meta = yaml.safe_load(f)

        img_path = yaml_path.parent / self.meta['image']
        raw = np.array(Image.open(img_path).convert('L'))
        self.resolution = self.meta['resolution']  # m/cell
        self.origin = self.meta['origin'][:2]       # [x, y] in metres

        # Nav2 convention: white (high pixel value) = free, black = occupied
        # Thresholds from the YAML (defaults match Nav2 map_server)
        free_thresh = self.meta.get('free_thresh', 0.196)
        occ_thresh = self.meta.get('occupied_thresh', 0.65)

        # Pixel value: 0 = black (occupied), 254 = white (free)
        # Nav2 uses: occ = (255 - pixel) / 255.  If occ > occ_thresh -> occupied
        #            if occ < free_thresh -> free
        occ_prob = (255.0 - raw.astype(float)) / 255.0
        self.grid = np.where(
            occ_prob < free_thresh, self.FREE,
            np.where(occ_prob > occ_thresh, self.OCCUPIED, self.UNKNOWN)
        ).astype(np.int8)

        self.height, self.width = self.grid.shape
        self.yaml_path = yaml_path

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (metres) to grid (row, col)."""
        c = int((x - self.origin[0]) / self.resolution)
        r = self.height - 1 - int((y - self.origin[1]) / self.resolution)
        return (r, c)

    def grid_to_world(self, r: int, c: int) -> Tuple[float, float]:
        """Convert grid (row, col) to world coordinates (metres)."""
        x = c * self.resolution + self.origin[0] + self.resolution / 2
        y = (self.height - 1 - r) * self.resolution + self.origin[1] + self.resolution / 2
        return (x, y)

    def place_obstacle(self, r: int, c: int, w: int, h: int,
                       grid: Optional[np.ndarray] = None) -> np.ndarray:
        """Place rectangular obstacle. Returns modified grid copy.

        Parameters
        ----------
        r, c : int
            Top-left corner in grid coordinates.
        w, h : int
            Width (columns) and height (rows) in cells.
        grid : np.ndarray, optional
            Grid to modify. Uses self.grid if None.
        """
        g = (grid if grid is not None else self.grid).copy()
        r_end = min(r + h, self.height)
        c_end = min(c + w, self.width)
        g[r:r_end, c:c_end] = self.OCCUPIED
        return g

    def is_free_rect(self, r: int, c: int, w: int, h: int,
                     grid: Optional[np.ndarray] = None) -> bool:
        """Check if a rectangular region is entirely free."""
        g = grid if grid is not None else self.grid
        if r < 0 or c < 0 or r + h > self.height or c + w > self.width:
            return False
        return np.all(g[r:r+h, c:c+w] == self.FREE)

    def save(self, grid: np.ndarray, output_stem: str):
        """Save modified grid as PGM + YAML for Nav2."""
        img = np.where(
            grid == self.FREE, 254,
            np.where(grid == self.OCCUPIED, 0, 205)
        ).astype(np.uint8)
        Image.fromarray(img).save(f"{output_stem}.pgm")
        meta = copy.deepcopy(self.meta)
        meta['image'] = f"{Path(output_stem).name}.pgm"
        with open(f"{output_stem}.yaml", 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)

    def summary(self) -> str:
        """Return a human-readable summary of the map."""
        n_free = int(np.sum(self.grid == self.FREE))
        n_occ = int(np.sum(self.grid == self.OCCUPIED))
        n_unk = int(np.sum(self.grid == self.UNKNOWN))
        total = self.height * self.width
        return (
            f"Map: {self.yaml_path.name}\n"
            f"  Size: {self.width}x{self.height} cells "
            f"({self.width * self.resolution:.1f}x"
            f"{self.height * self.resolution:.1f} m)\n"
            f"  Resolution: {self.resolution} m/cell\n"
            f"  Origin: ({self.origin[0]:.2f}, {self.origin[1]:.2f})\n"
            f"  Free: {n_free} ({100*n_free/total:.1f}%), "
            f"Occupied: {n_occ} ({100*n_occ/total:.1f}%), "
            f"Unknown: {n_unk} ({100*n_unk/total:.1f}%)"
        )
