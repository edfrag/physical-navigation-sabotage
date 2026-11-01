"""Re-run maze and hospital at inflation_radius=6 (inscribed_radius ~0.18m).

This approximates Nav2's truly impassable zone. Cells within inscribed_radius
are lethal in Nav2; beyond that is soft cost decay (not a barrier).

Saves to:
  results/offline_results_inflation6.json  — main sweep
  results/betweenness_inflation6.json
  results/submodularity_inflation6_<map>.json
  results/multi_route_hospital.json        — Priority 1 for hospital
  results/multi_route_maze.json            — Priority 1 for maze
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import astar, path_cost, inflate
from src.analysis import compute_submodularity_ratio, marginal_gain_analysis

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

INFLATION = 6  # inscribed_radius equivalent
N_RANDOM_SEEDS = 10


def find_valid(omap, pt, inf_r):
    """Find nearest valid (non-inflated) cell to pt in world coords."""
    inf_grid = inflate(omap.grid, inf_r)
    if inf_grid[pt[0], pt[1]] < 99:
        return pt
    for d in range(1, 100):
        for dr in range(-d, d+1):
            for dc in range(-d, d+1):
                nr, nc = pt[0]+dr, pt[1]+dc
                if 0<=nr<omap.height and 0<=nc<omap.width and inf_grid[nr,nc]<99:
                    return (nr, nc)
    return None


MAPS = {
    'maze': {
        'yaml': 'maps/maze.yaml', 'res': 0.03,
        'start': (-7.535, -7.855), 'goal': (6.775, 7.085),
        'min_dist': 30, 'obj_cells': 10,
    },
    'hospital': {
        'yaml': 'maps/hospital.yaml', 'res': 0.05,
        'start': (1.525, 12.125), 'goal': (53.525, 12.125),
        'min_dist': 20, 'obj_cells': 6,
    },
}


def make_solver(name, cfg, k):
    omap = OccupancyMap(cfg['yaml'])
    # Auto-adjust start/goal for the inflation value
    s_grid = omap.world_to_grid(*cfg['start'])
    g_grid = omap.world_to_grid(*cfg['goal'])
    s_valid = find_valid(omap, s_grid, INFLATION)
    g_valid = find_valid(omap, g_grid, INFLATION)
    s_world = omap.grid_to_world(*s_valid)
    g_world = omap.grid_to_world(*g_valid)
    return AOPPSolver(
        omap, s_world, g_world,
        k=k, obj_w=cfg['obj_cells'], obj_h=cfg['obj_cells'],
        corridor_d=10, inflation_r=INFLATION,
        candidate_stride=2,
        min_dist_to_endpoints=cfg['min_dist'],
    ), omap


def main_sweep():
    out = RESULTS_DIR / 'offline_results_inflation6.json'
    sub_out = {}
    if out.exists():
        print(f"[A] Main sweep inflation=6 — SKIP (exists)")
        return

    print(f"\n{'='*60}")
    print(f"MAIN SWEEP at inflation_radius={INFLATION}")
    print(f"{'='*60}")

    results = []
    for name, cfg in MAPS.items():
        for k in [1, 2, 3, 5, 7]:
            print(f"\n--- {name} k={k} (inflation={INFLATION}) ---")
            solver, omap = make_solver(name, cfg, k)
            g_place, g_gains, g_grid = solver.solve()
            g_delay = sum(g_gains)
            pb_place, pb_delay, _ = solver.pathblock_placement()
            r_delays = [solver.random_placement(seed=s)[1] for s in range(N_RANDOM_SEEDS)]
            pct = g_delay / solver.baseline_cost * 100 if g_delay < 1e6 else None

            results.append({
                'env': name, 'k': k, 'inflation_radius': INFLATION,
                'baseline_cost': solver.baseline_cost,
                'greedy_delay': g_delay,
                'greedy_delay_pct': pct,
                'is_denial': g_delay >= 1e6,
                'greedy_placements': [(int(r), int(c)) for r, c in g_place],
                'greedy_marginal_gains': g_gains,
                'pathblock_delay': pb_delay,
                'random_delay_mean': float(np.mean(r_delays)),
                'random_delay_std': float(np.std(r_delays)),
            })
            status = "DENIAL" if g_delay >= 1e6 else f"{g_delay:.1f} ({pct:.1f}%)"
            print(f"  greedy={status}  pathblock={pb_delay:.1f}  random={np.mean(r_delays):.1f}+/-{np.std(r_delays):.1f}")

        # Submodularity
        print(f"\n--- {name} submodularity ---")
        solver, _ = make_solver(name, cfg, 5)
        p_sub, g_sub, _ = solver.solve(verbose=False)
        gamma, n_comp, _ = compute_submodularity_ratio(
            OccupancyMap(cfg['yaml']), solver.start, solver.goal,
            p_sub, solver.candidates[:20],
            inflation_r=INFLATION, obj_w=cfg['obj_cells'], obj_h=cfg['obj_cells'],
        )
        mg = marginal_gain_analysis(g_sub)
        sub_entry = {
            'env': name, 'gamma': float(gamma) if not np.isnan(gamma) else None,
            'n_comparisons': n_comp, 'monotone_decreasing': mg['monotone_decreasing'],
        }
        sub_out[name] = sub_entry
        with open(RESULTS_DIR / f'submodularity_inflation6_{name}.json', 'w') as f:
            json.dump(sub_entry, f, indent=2)
        print(f"  gamma={sub_entry['gamma']}, monotone={sub_entry['monotone_decreasing']}")

    with open(out, 'w') as f:
        json.dump({'results': results, 'submodularity': sub_out,
                   'inflation_radius': INFLATION}, f, indent=2, default=str)
    print(f"\nSaved: {out}")


# ============================================================
# Betweenness baseline at inflation=6
# ============================================================
def compute_betweenness(omap, inflation_r, n_paths=200):
    grid = omap.grid
    free_r, free_c = np.where(grid == 0)
    n_free = len(free_r)
    cell_counts = np.zeros(grid.shape, dtype=np.int32)
    rng = np.random.RandomState(42)
    computed = 0
    for _ in range(n_paths * 5):
        if computed >= n_paths: break
        i_s, i_g = rng.randint(n_free), rng.randint(n_free)
        s = (int(free_r[i_s]), int(free_c[i_s]))
        g = (int(free_r[i_g]), int(free_c[i_g]))
        if abs(s[0]-g[0]) + abs(s[1]-g[1]) < 50: continue
        p = astar(grid, s, g, inflation_r)
        if p is None: continue
        for (r, c) in p: cell_counts[r, c] += 1
        computed += 1
    return cell_counts


def betweenness_placement(omap, cell_counts, start, goal, k, obj_w, obj_h,
                          min_dist, inflation_r):
    grid = omap.grid.copy()
    counts = cell_counts.copy()
    placements = []
    for i in range(k):
        best_score = -1; best_pos = None
        for r in range(omap.height - obj_h):
            for c in range(omap.width - obj_w):
                if not omap.is_free_rect(r, c, obj_w, obj_h, grid): continue
                ds = max(abs(r - start[0]), abs(c - start[1]))
                dg = max(abs(r - goal[0]), abs(c - goal[1]))
                if ds < min_dist or dg < min_dist: continue
                score = int(counts[r:r+obj_h, c:c+obj_w].sum())
                if score > best_score: best_score = score; best_pos = (r, c)
        if best_pos is None: break
        grid = omap.place_obstacle(best_pos[0], best_pos[1], obj_w, obj_h, grid)
        counts[best_pos[0]:best_pos[0]+obj_h, best_pos[1]:best_pos[1]+obj_w] = 0
        placements.append(best_pos)
    p = astar(grid, start, goal, inflation_r)
    baseline = path_cost(astar(omap.grid, start, goal, inflation_r))
    cost = path_cost(p)
    delay = min(cost - baseline, 1e6) if cost != float('inf') else 1e6
    return placements, delay


def betweenness_sweep():
    out = RESULTS_DIR / 'betweenness_inflation6.json'
    if out.exists():
        print(f"[B] Betweenness inflation=6 — SKIP (exists)")
        return

    print(f"\n{'='*60}")
    print(f"BETWEENNESS BASELINE at inflation={INFLATION}")
    print(f"{'='*60}")

    results = []
    for name, cfg in MAPS.items():
        print(f"\n--- {name} ---")
        omap = OccupancyMap(cfg['yaml'])
        s_grid = omap.world_to_grid(*cfg['start'])
        g_grid = omap.world_to_grid(*cfg['goal'])
        s = find_valid(omap, s_grid, INFLATION)
        g = find_valid(omap, g_grid, INFLATION)
        oc = cfg['obj_cells']
        cell_counts = compute_betweenness(omap, INFLATION)
        print(f"  Betweenness computed (max={cell_counts.max()})")

        for k in [1, 2, 3, 5]:
            placements, delay = betweenness_placement(
                omap, cell_counts, s, g, k, oc, oc,
                cfg['min_dist'], INFLATION)
            results.append({
                'env': name, 'k': k, 'inflation_radius': INFLATION,
                'betweenness_delay': delay,
                'betweenness_placements': [(int(r), int(c)) for r, c in placements],
            })
            print(f"  k={k}: betweenness_delay={delay:.1f}")

    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out}")


# ============================================================
# Multi-route experiments (Priority 1)
# ============================================================

MAZE_ROUTES = [
    {'name': 'sw_to_ne', 'start': (-7.535, -7.855), 'goal': (6.775, 7.085)},
    {'name': 'nw_to_se', 'start': (-7.0, 7.0), 'goal': (7.0, -7.0)},
    {'name': 'center_span', 'start': (-6.0, 0.0), 'goal': (6.0, 0.0)},
]

HOSPITAL_ROUTES = [
    # Long corridor end-to-end
    {'name': 'corridor_endtoend', 'start': (1.525, 12.125), 'goal': (53.525, 12.125)},
    # Cross-wing through central junction (top room left wing to top room right wing)
    {'name': 'cross_wing_top', 'start': (5.5, 18.0), 'goal': (48.0, 18.0)},
    # Short intra-wing (within left wing, room-to-room)
    {'name': 'intra_left_wing', 'start': (3.0, 18.0), 'goal': (13.0, 6.0)},
    # Nurse station area to far room in right wing
    {'name': 'nurse_to_far_room', 'start': (28.0, 18.0), 'goal': (51.0, 6.0)},
]


def multi_route(name, cfg, routes, out_path):
    """Run multi-route with per-(route, k) checkpointing.

    Skips any (route, k) pair that's already in the output file.
    """
    print(f"\n{'='*60}")
    print(f"MULTI-ROUTE: {name} at inflation={INFLATION}")
    print(f"{'='*60}")

    # Load existing results if any
    results = []
    if Path(out_path).exists():
        results = json.load(open(out_path))
        done_pairs = {(r.get('route'), r.get('k')) for r in results}
        print(f"  Resuming: {len(done_pairs)} (route, k) pairs already done")

    def is_done(route_name, k):
        return any(r.get('route') == route_name and r.get('k') == k for r in results)

    omap = OccupancyMap(cfg['yaml'])
    for route in routes:
        # Adjust start/goal
        s_grid = omap.world_to_grid(*route['start'])
        g_grid = omap.world_to_grid(*route['goal'])
        s_v = find_valid(omap, s_grid, INFLATION)
        g_v = find_valid(omap, g_grid, INFLATION)
        if not s_v or not g_v:
            print(f"\n--- {name} route '{route['name']}' — SKIP (no valid start/goal)")
            continue
        sw = omap.grid_to_world(*s_v)
        gw = omap.grid_to_world(*g_v)

        # Check baseline
        baseline_path = astar(omap.grid, s_v, g_v, INFLATION)
        if baseline_path is None:
            if not any(r.get('route') == route['name'] and r.get('note') for r in results):
                results.append({
                    'route': route['name'], 'start_world': sw, 'goal_world': gw,
                    'baseline_cost': None, 'note': 'no baseline path'})
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            print(f"\n--- {name} route '{route['name']}' — NO BASELINE PATH")
            continue

        for k in [5, 7]:
            if is_done(route['name'], k):
                existing = next(r for r in results if r.get('route') == route['name'] and r.get('k') == k)
                d = existing.get('greedy_delay')
                status = "DENIAL" if d and d >= 1e6 else f"{d:.1f}" if d else "error"
                print(f"\n--- {name} route '{route['name']}' k={k} — SKIP (done: {status})")
                continue

            print(f"\n--- {name} route '{route['name']}' k={k} ---")
            print(f"  Baseline: {path_cost(baseline_path):.0f}")
            try:
                solver = AOPPSolver(
                    omap, sw, gw,
                    k=k, obj_w=cfg['obj_cells'], obj_h=cfg['obj_cells'],
                    corridor_d=10, inflation_r=INFLATION,
                    candidate_stride=2,
                    min_dist_to_endpoints=cfg['min_dist'],
                )
                g_place, g_gains, _ = solver.solve(verbose=False)
                g_delay = sum(g_gains)
                pct = g_delay / solver.baseline_cost * 100 if g_delay < 1e6 else None
                status = "DENIAL" if g_delay >= 1e6 else f"{g_delay:.1f} ({pct:.1f}%)"
                print(f"  k={k}: {status}")
                results.append({
                    'route': route['name'], 'k': k,
                    'start_world': sw, 'goal_world': gw,
                    'baseline_cost': solver.baseline_cost,
                    'greedy_delay': g_delay,
                    'greedy_delay_pct': pct,
                    'is_denial': g_delay >= 1e6,
                    'placements': [(int(r), int(c)) for r, c in g_place],
                })
            except Exception as e:
                print(f"  k={k}: ERROR {e}")
                results.append({'route': route['name'], 'k': k, 'error': str(e)})

            # Checkpoint after each (route, k) pair
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  [checkpoint saved]")

    print(f"All routes done for {name}")


if __name__ == '__main__':
    main_sweep()
    betweenness_sweep()
    multi_route('maze', MAPS['maze'], MAZE_ROUTES,
                RESULTS_DIR / 'multi_route_maze.json')
    multi_route('hospital', MAPS['hospital'], HOSPITAL_ROUTES,
                RESULTS_DIR / 'multi_route_hospital.json')
    print("\n" + "="*60)
    print("ALL INFLATION=6 EXPERIMENTS COMPLETE")
    print("="*60)
