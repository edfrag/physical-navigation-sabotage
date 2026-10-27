#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from aopp.baselines import pathblock_attack, random_attack
from aopp.config import load_config
from aopp.greedy import greedy_attack
from aopp.grid import load_occupancy
from aopp.inflation import inflate
from aopp.planning import shortest_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["greedy", "random", "pathblock"], default="greedy")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    occ = load_occupancy(cfg.map_path)
    baseline, _ = shortest_path(inflate(occ, cfg.inflation_radius_cells), cfg.start, cfg.goal)
    if math.isinf(baseline):
        raise SystemExit("baseline route is infeasible")

    if args.method == "greedy":
        result = greedy_attack(
            occ, cfg.start, cfg.goal, cfg.footprint_size_cells,
            cfg.inflation_radius_cells, cfg.stealth_distance_cells,
            args.k, cfg.candidate_stride, cfg.candidate_buffer_cells,
        )
        placements = result.placements
        attacked = result.final_cost
        step_rows = [s.__dict__ for s in result.steps]
    elif args.method == "random":
        placements, attacked = random_attack(
            occ, cfg.start, cfg.goal, cfg.footprint_size_cells,
            cfg.inflation_radius_cells, cfg.stealth_distance_cells, args.k, args.seed,
        )
        step_rows = []
    else:
        placements, attacked = pathblock_attack(
            occ, cfg.start, cfg.goal, cfg.footprint_size_cells,
            cfg.inflation_radius_cells, cfg.stealth_distance_cells, args.k,
        )
        step_rows = []

    delay = math.inf if math.isinf(attacked) else attacked - baseline
    payload = {
        "method": args.method,
        "k": args.k,
        "seed": args.seed if args.method == "random" else None,
        "baseline_cost_cells": baseline,
        "attacked_cost_cells": None if math.isinf(attacked) else attacked,
        "delay_cells": None if math.isinf(delay) else delay,
        "denial": math.isinf(attacked),
        "placements": placements,
        "steps": step_rows,
    }

    def sanitize(value):
        if isinstance(value, float) and math.isinf(value):
            return None
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(v) for v in value]
        return value

    encoded = json.dumps(sanitize(payload), indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
