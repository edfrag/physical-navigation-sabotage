"""Stride=4 calibration on 5 Level 1 stratified routes.

Re-runs greedy k=5 at candidate_stride=4 on 5 routes that were already
evaluated at stride=6, then compares delay/denial outcomes.

Output: results/northern_private/level1_stride4_calibration/route_sweep.json
        results/northern_private/level1_stride4_calibration/comparison.json
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
CANDIDATE_STRIDE = 4
MIN_DIST = 15

OUT_DIR = Path('results/northern_private/level1_stride4_calibration')
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def main():
    # Load the existing stride=6 stratified results (top of selection)
    s6 = json.load(open('results/northern_private/level1_route_sweep/route_sweep.json'))
    # Pick the first 5 successfully-completed routes (in selection order)
    s6_routes = [r for r in s6['results'] if 'compute_time_s' in r][:5]
    print(f'Calibrating against {len(s6_routes)} stride=6 routes')

    # Load candidate catalog to get start/goal
    candidates = json.load(open('results/northern_private/level1_routes.json'))
    cand_by_id = {c['candidate_id']: c for c in candidates}

    omap = OccupancyMap('maps/level1.yaml')
    out_path = OUT_DIR / 'route_sweep.json'
    results = []
    if out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get('results', [])
        done_ids = {r.get('candidate_id') for r in results}
        print(f'Resuming with {len(done_ids)} done')
    else:
        done_ids = set()

    for s6_r in s6_routes:
        cid = s6_r['candidate_id']
        if cid in done_ids:
            print(f'\nSKIP cand {cid} (already done)')
            continue
        cand = cand_by_id[cid]
        sw = tuple(cand['start_world'])
        gw = tuple(cand['goal_world'])
        print(f'\n=== cand {cid} (baseline {cand["baseline_length_m"]:.1f}m) ===')

        t0 = time.time()
        try:
            solver = AOPPSolver(
                omap, sw, gw,
                k=K, obj_w=OBJ_CELLS, obj_h=OBJ_CELLS,
                corridor_d=10, inflation_r=INFLATION,
                candidate_stride=CANDIDATE_STRIDE,
                min_dist_to_endpoints=MIN_DIST,
                n_workers=N_WORKERS,
            )
            place, gains, _ = solver.solve(verbose=False)
            delay = sum(gains)
            is_denial = delay >= 1e6
            elapsed = time.time() - t0
            pct = (delay / solver.baseline_cost * 100) if not is_denial else None
            tag = 'DENIAL' if is_denial else f'{delay:.1f} ({pct:.1f}%)'
            print(f'  greedy stride=4: {tag}  ({elapsed:.0f}s)')
            results.append({
                'candidate_id': cid,
                'baseline_length_m': cand['baseline_length_m'],
                'baseline_cost': solver.baseline_cost,
                'k': K,
                'stride': CANDIDATE_STRIDE,
                'greedy_delay': delay,
                'greedy_delay_pct': pct,
                'is_denial': bool(is_denial),
                'placements': [(int(r), int(c)) for r, c in place],
                'compute_time_s': float(elapsed),
                # Cross-reference stride=6 numbers for direct comparison
                'stride6_delay': s6_r.get('greedy_delay'),
                'stride6_delay_pct': s6_r.get('greedy_delay_pct'),
                'stride6_is_denial': s6_r.get('is_denial'),
                'stride6_compute_time_s': s6_r.get('compute_time_s'),
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f'  ERROR: {type(e).__name__}: {e}')
            results.append({'candidate_id': cid, 'error': str(e),
                            'compute_time_s': float(elapsed)})

        with open(out_path, 'w') as f:
            json.dump({
                'config': {'inflation': INFLATION, 'k': K,
                           'obj_cells': OBJ_CELLS,
                           'candidate_stride': CANDIDATE_STRIDE,
                           'min_dist': MIN_DIST,
                           'rationale': (
                               'Calibration: re-run 5 stride=6 stratified L1 routes '
                               'at stride=4 to check whether stride choice materially '
                               'changes greedy delay/denial outcomes. '
                               'Reviewer-defense Q3.'
                           )},
                'results': results,
            }, f, indent=2, default=str)

    # Generate comparison summary
    comparison_path = OUT_DIR / 'comparison.json'
    summary = {
        'n_routes': len([r for r in results if 'greedy_delay' in r]),
        'agreements': {
            'denial_match': 0, 'denial_mismatch': 0,
            'delay_pct_within_5': 0, 'delay_pct_within_10': 0,
            'delay_pct_diverge_>10': 0,
        },
        'per_route': [],
    }
    for r in results:
        if 'greedy_delay' not in r:
            continue
        s4d = r.get('greedy_delay_pct') or 0
        s6d = r.get('stride6_delay_pct') or 0
        d4 = r.get('is_denial')
        d6 = r.get('stride6_is_denial')
        diff = abs(s4d - s6d)
        if d4 == d6:
            summary['agreements']['denial_match'] += 1
        else:
            summary['agreements']['denial_mismatch'] += 1
        if diff <= 5:
            summary['agreements']['delay_pct_within_5'] += 1
        elif diff <= 10:
            summary['agreements']['delay_pct_within_10'] += 1
        else:
            summary['agreements']['delay_pct_diverge_>10'] += 1
        summary['per_route'].append({
            'candidate_id': r['candidate_id'],
            'baseline_m': r['baseline_length_m'],
            'stride4_delay_pct': s4d, 'stride6_delay_pct': s6d,
            'abs_diff_pct': diff,
            'denial_match': d4 == d6,
        })
    comparison_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nSaved {out_path}')
    print(f'Saved {comparison_path}')
    print(f'\n=== Calibration verdict ===')
    a = summary['agreements']
    print(f'  Denial outcome match: {a["denial_match"]}/{summary["n_routes"]}')
    print(f'  Delay % within 5pp: {a["delay_pct_within_5"]}')
    print(f'  Delay % within 10pp: {a["delay_pct_within_10"]}')
    print(f'  Delay % diverge >10pp: {a["delay_pct_diverge_>10"]}')


if __name__ == '__main__':
    main()
