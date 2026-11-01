"""Topology metric: per-route deniability margin m_route.

For each route, find the max MIN_DIST (cells, distance from endpoints in which
the attacker may NOT place objects) at which greedy k=5 still produces a
denial under the offline binary surrogate. Larger m_route = more
topologically vulnerable (a deeper bottleneck the attacker can hit even
with severe stealth constraints).

Routes:
  - 30 from results/additional_experiments/route_sweep/route_sweep.json
  - 4 named paper routes from exp2b/placements.json:
    corridor_endtoend, nurse_to_far_room, auto_route_2, auto_route_20

Method: binary search over MIN_DIST grid {5,10,15,20,25,30,40,50,75,100}.
       For each candidate MIN_DIST, run greedy k=5 and check denial via
       sum(marginal_gains) >= 1e6 (the same offline surrogate criterion
       used by AOPPSolver and the route_sweep).

Output: results/route_topology_metric/topology_metric.json
       (resumable; one entry per route)

Note on denial method: the offline A* surrogate saturates at k=1 across all
30 hospital sweep routes. Nav2 NavFn may differ (corridor_endtoend is the
known counterexample). m_route here is the surrogate margin; ground-truth
NavFn margin requires Gazebo trials.

Usage:
  .venv/bin/python3 -m experiments.exp_topology_metric
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.map_utils import OccupancyMap
from src.aopp_solver import AOPPSolver
from src.planner import astar, inflate

INFLATION = 6
K = 5
OBJ_CELLS = 6
N_WORKERS = 24
CANDIDATE_STRIDE = 8
MIN_DIST_GRID = [15, 30, 50, 100]

OUT_DIR = Path('results/route_topology_metric')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'topology_metric.json'

HOSPITAL_YAML = 'maps/hospital.yaml'
SWEEP_PATH = 'results/additional_experiments/route_sweep/route_sweep.json'
E2B_PATH = 'results/additional_experiments/exp2b/placements.json'


def find_valid(omap, pt, inf_r):
    inf_grid = inflate(omap.grid, inf_r)
    if inf_grid[pt[0], pt[1]] < 99:
        return pt
    for d in range(1, 50):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = pt[0] + dr, pt[1] + dc
                if (0 <= nr < omap.height and 0 <= nc < omap.width
                        and inf_grid[nr, nc] < 99):
                    return (nr, nc)
    return pt


def run_eval(omap, sw, gw, min_dist):
    """Run greedy k=K at given MIN_DIST. Return (is_denial, total_delay,
    elapsed_seconds, n_candidates). is_denial=False if no candidates."""
    t0 = time.time()
    try:
        solver = AOPPSolver(
            omap, sw, gw,
            k=K, obj_w=OBJ_CELLS, obj_h=OBJ_CELLS,
            corridor_d=10, inflation_r=INFLATION,
            candidate_stride=CANDIDATE_STRIDE,
            min_dist_to_endpoints=min_dist,
            n_workers=N_WORKERS,
        )
        # Empty candidate set?
        n_cand = len(getattr(solver, 'candidates', []))
        if n_cand == 0:
            return False, 0.0, time.time() - t0, 0
        place, gains, _ = solver.solve(verbose=False)
        total = sum(gains)
        is_denial = total >= 1e6
        return is_denial, float(total), time.time() - t0, n_cand
    except Exception as e:
        return False, 0.0, time.time() - t0, -1


def find_m_route(omap, sw, gw, route_id, log):
    """Binary search over MIN_DIST_GRID for the largest value where
    denial holds. Returns (m_route_cells, sweep_results).
    m_route = 0 if even MIN_DIST_GRID[0] doesn't deny.
    m_route >= MIN_DIST_GRID[-1] if even the largest denies."""
    g = MIN_DIST_GRID
    sweep = {}  # min_dist -> {is_denial, delay, time_s, n_cand}

    def eval_at(i):
        if g[i] in sweep:
            return sweep[g[i]]['is_denial']
        d, total, t, n = run_eval(omap, sw, gw, g[i])
        sweep[g[i]] = {'is_denial': d, 'total_delay': total,
                       'time_s': round(t, 1), 'n_candidates': n}
        log(f"    MIN_DIST={g[i]:3d} cand={n:5d} denial={d} delay={total:.0f} ({t:.0f}s)")
        return d

    # First test largest: if denies, m_route is at least MIN_DIST_GRID[-1]
    if eval_at(len(g) - 1):
        return g[-1], sweep, True
    # Test smallest: if doesn't deny, m_route < MIN_DIST_GRID[0]
    if not eval_at(0):
        return 0, sweep, False
    # Bisect: invariant: eval_at(lo)=True, eval_at(hi)=False
    lo, hi = 0, len(g) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if eval_at(mid):
            lo = mid
        else:
            hi = mid
    return g[lo], sweep, False


def collect_routes():
    """Build the 34-route inventory."""
    omap = OccupancyMap(HOSPITAL_YAML)
    routes = []

    sweep = json.load(open(SWEEP_PATH))
    for r in sweep['results']:
        if r.get('error') or 'placements' not in r:
            continue
        routes.append({
            'route_id': r['route'],
            'source': 'sweep',
            'candidate_id': r.get('candidate_id'),
            'start_world': r['start_world'],
            'goal_world': r['goal_world'],
            'baseline_cost': r['baseline_cost'],
        })

    e2b = json.load(open(E2B_PATH))
    for r in e2b:
        if r.get('error') or not r.get('placements'):
            continue
        s_grid = omap.world_to_grid(*r['start_world'])
        g_grid = omap.world_to_grid(*r['goal_world'])
        s_cell = find_valid(omap, s_grid, INFLATION)
        g_cell = find_valid(omap, g_grid, INFLATION)
        baseline = astar(omap.grid, s_cell, g_cell, INFLATION)
        baseline_cost = len(baseline) if baseline else None
        routes.append({
            'route_id': r['route'],
            'source': 'exp2b',
            'candidate_id': None,
            'start_world': r['start_world'],
            'goal_world': r['goal_world'],
            'baseline_cost': baseline_cost,
        })
    return omap, routes


def main():
    omap, routes = collect_routes()
    print(f"Total routes: {len(routes)} ({sum(1 for r in routes if r['source']=='sweep')} sweep + "
          f"{sum(1 for r in routes if r['source']=='exp2b')} named)")

    # Resume from previous run
    results = []
    done = set()
    if OUT.exists():
        prev = json.load(open(OUT))
        results = prev.get('results', [])
        done = {r['route_id'] for r in results}
        print(f"Resuming with {len(done)} routes done")

    log_lines = []
    def log(s):
        print(s)
        log_lines.append(s)

    t_start = time.time()
    for i, route in enumerate(routes):
        rid = route['route_id']
        if rid in done:
            print(f"\n[{i+1}/{len(routes)}] {rid}: SKIP (done)")
            continue
        sw = tuple(route['start_world'])
        gw = tuple(route['goal_world'])
        baseline_cost = route['baseline_cost']
        print(f"\n[{i+1}/{len(routes)}] {rid} (source={route['source']}, baseline={baseline_cost} cells):")
        m_cells, sweep, at_max = find_m_route(omap, sw, gw, rid, log)
        m_m = round(m_cells * 0.05, 3) if m_cells > 0 else 0.0
        denial_at_default = sweep.get(15, {}).get('is_denial', None)
        if denial_at_default is None:
            # We may have skipped MIN_DIST=15 in binary search; infer it
            denial_at_default = m_cells >= 15
        entry = {
            'route_id': rid,
            'source': route['source'],
            'candidate_id': route.get('candidate_id'),
            'baseline_path_len_cells': baseline_cost,
            'baseline_path_len_m': round((baseline_cost or 0) * 0.05, 2),
            'm_route_cells': int(m_cells),
            'm_route_m': float(m_m),
            'm_route_at_grid_max': bool(at_max),
            'denial_at_default_min_dist': bool(denial_at_default),
            'min_dist_grid': MIN_DIST_GRID,
            'sweep': sweep,
        }
        results.append(entry)
        with open(OUT, 'w') as f:
            json.dump({
                'config': {
                    'inflation': INFLATION, 'k': K, 'obj_cells': OBJ_CELLS,
                    'candidate_stride': CANDIDATE_STRIDE,
                    'denial_method': 'binary_surrogate_astar',
                    'min_dist_grid': MIN_DIST_GRID,
                    'note': 'm_route uses offline A* surrogate; Nav2 NavFn '
                            'ground truth may differ (corridor_endtoend is the '
                            'documented surrogate-vs-NavFn counterexample).',
                },
                'results': results,
            }, f, indent=2, default=str)
        elapsed = time.time() - t_start
        n_done = len([r for r in results if r['route_id'] not in done])
        if n_done > 0:
            avg = elapsed / n_done
            remaining = (len(routes) - i - 1) * avg
            print(f"  -> m_route = {m_cells} cells ({m_m} m). "
                  f"Elapsed {elapsed/60:.1f} min, ~{remaining/60:.0f} min left")

    # Summary
    print("\n" + "=" * 60)
    print(f"Done. {len(results)} routes evaluated.")
    ms = [r['m_route_cells'] for r in results]
    if ms:
        import statistics as st
        print(f"m_route distribution (cells):")
        print(f"  min={min(ms)}, Q1={int(st.quantiles(ms, n=4)[0])}, "
              f"median={int(st.median(ms))}, Q3={int(st.quantiles(ms, n=4)[2])}, "
              f"max={max(ms)}")


if __name__ == '__main__':
    main()
