from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import yaml

Cell = Tuple[int, int]


@dataclass(frozen=True)
class ExperimentConfig:
    map_path: Path
    resolution_m: float
    inflation_radius_cells: int
    footprint_size_cells: int
    start: Cell
    goal: Cell
    stealth_distance_cells: float
    candidate_stride: int = 1
    candidate_buffer_cells: int = 2


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    map_path = Path(raw["map"])
    if not map_path.is_absolute():
        map_path = (path.parent / map_path).resolve()
    return ExperimentConfig(
        map_path=map_path,
        resolution_m=float(raw["resolution_m"]),
        inflation_radius_cells=int(raw["inflation_radius_cells"]),
        footprint_size_cells=int(raw["footprint_size_cells"]),
        start=tuple(map(int, raw["start"])),
        goal=tuple(map(int, raw["goal"])),
        stealth_distance_cells=float(raw["stealth_distance_cells"]),
        candidate_stride=int(raw.get("candidate_stride", 1)),
        candidate_buffer_cells=int(raw.get("candidate_buffer_cells", 2)),
    )
