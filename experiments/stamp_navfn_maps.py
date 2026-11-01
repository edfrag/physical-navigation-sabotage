"""Pre-stamp PGM maps for NavFn evaluation.

For each (env, route, method, k) of interest, produce a modified PGM (with
attack boxes stamped as occupied) plus a YAML, and a manifest listing all
stamped maps. The actual NavFn evaluation runs inside the Docker container
(scripts/nav2_planner_only.py) reading the manifest.

Sources:
  - results/topology_baselines/baselines_main.json (greedy + 3 baselines x 4 envs x 5 k)
  - results/additional_experiments/route_sweep/route_sweep.json (30 hospital sweep routes, k=5)
  - results/additional_experiments/exp2b/placements.json (4 named hospital routes, k=5)

Output:
  maps/navfn_eval/<env>_<route>_<method>_k<k>.{pgm,yaml}
  results/topology_baselines/navfn_manifest.json
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

OUT_DIR = Path('maps/navfn_eval')
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = Path('results/topology_baselines/navfn_manifest.json')

ENV_YAML = {
    'maze': 'maps/maze.yaml',
    'hospital': 'maps/hospital.yaml',
    'warehouse': 'maps/warehouse.yaml',
    'depot': 'maps/depot.yaml',
}
ENV_OBJ = {'maze': 10, 'hospital': 6, 'warehouse': 10, 'depot': 8}


def stamp_pgm(src_yaml: Path, placements_grid_topleft, obj_cells, out_name):
    with open(src_yaml) as f:
        meta = yaml.safe_load(f)
    img = np.array(Image.open(src_yaml.parent / meta['image']).convert('L'))
    h, w = img.shape
    for (r, c) in placements_grid_topleft:
        # PGM rows are top-down (image origin top-left); our internal grid
        # uses (row, col) consistent with the OccupancyMap convention.
        # We need to stamp at the same row indexing the loader uses.
        # OccupancyMap.world_to_grid sets row = (h-1) - cell.row in load,
        # so the placement row from the optimizer is already image-row-flipped.
        # Stamp directly:
        r0 = max(0, r); r1 = min(h, r + obj_cells)
        c0 = max(0, c); c1 = min(w, c + obj_cells)
        img[r0:r1, c0:c1] = 0  # 0 = occupied in PGM convention
    out_pgm = OUT_DIR / f'{out_name}.pgm'
    out_yaml = OUT_DIR / f'{out_name}.yaml'
    Image.fromarray(img).save(out_pgm)
    new_meta = dict(meta)
    new_meta['image'] = str(out_pgm.name)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(new_meta, f)
    return str(out_pgm), str(out_yaml)


def main():
    manifest = []

    # Item A: greedy + 3 baselines on 4 envs at k ∈ {1..7}
    p1 = Path('results/topology_baselines/baselines_main.json')
    if p1.exists():
        d = json.load(open(p1))
        for r in d.get('results', []):
            env = r['env']; method = r['method']; k = r['k']
            placements = r.get('placements', [])
            if not placements:
                continue
            obj = ENV_OBJ.get(env, 6)
            name = f'{env}_canonical_{method}_k{k}'
            try:
                pgm, yml = stamp_pgm(
                    Path(ENV_YAML[env]), placements, obj, name)
                manifest.append({
                    'env': env, 'route': 'canonical', 'method': method, 'k': k,
                    'pgm': pgm, 'yaml': yml, 'placements_grid': placements,
                    'source': 'topology_baselines',
                })
            except Exception as e:
                print(f'Failed: {name}: {e}')

    # Hospital sweep routes (k=5 greedy)
    p2 = Path('results/additional_experiments/route_sweep/route_sweep.json')
    if p2.exists():
        d = json.load(open(p2))
        for r in d.get('results', []):
            if r.get('error') or not r.get('placements'):
                continue
            placements = r['placements']
            obj = ENV_OBJ['hospital']
            name = f'hospital_{r["route"]}_greedy_k5'
            try:
                pgm, yml = stamp_pgm(
                    Path(ENV_YAML['hospital']), placements, obj, name)
                manifest.append({
                    'env': 'hospital', 'route': r['route'],
                    'method': 'greedy', 'k': 5,
                    'pgm': pgm, 'yaml': yml,
                    'placements_grid': placements,
                    'start_world': r.get('start_world'),
                    'goal_world': r.get('goal_world'),
                    'source': 'route_sweep',
                })
            except Exception as e:
                print(f'Failed: {name}: {e}')

    # NPH stratified-route greedy placements
    for floor in ['level1', 'level3']:
        sweep_path = Path(f'results/northern_private/{floor}_route_sweep/route_sweep.json')
        if not sweep_path.exists():
            continue
        d = json.load(open(sweep_path))
        for r in d.get('results', []):
            if r.get('error') or not r.get('placements'):
                continue
            placements = r['placements']
            ENV_YAML[floor] = f'maps/{floor}.yaml'
            obj = ENV_OBJ['hospital']
            name = f'{floor}_{r["route"]}_greedy_k5'
            try:
                pgm, yml = stamp_pgm(
                    Path(ENV_YAML[floor]), placements, obj, name)
                manifest.append({
                    'env': floor, 'route': r['route'],
                    'method': 'greedy', 'k': 5,
                    'pgm': pgm, 'yaml': yml,
                    'placements_grid': placements,
                    'start_world': r.get('start_world'),
                    'goal_world': r.get('goal_world'),
                    'source': 'nph_stratified',
                })
            except Exception as e:
                print(f'Failed: {name}: {e}')

    # NPH named-route placements: 8 routes (4 per floor) x 4 attackers
    for floor in ['level1', 'level3']:
        nb_path = Path(f'results/northern_private/{floor}_named_baselines.json')
        if not nb_path.exists():
            continue
        d = json.load(open(nb_path))
        for r in d.get('results', []):
            if r.get('error') or not r.get('placements'):
                continue
            placements = r['placements']
            obj = ENV_OBJ['hospital']  # NPH uses obj_w=6 like existing hospital
            method = r['method']
            rid = r['route_id']
            # NPH map yamls are at maps/{floor}.yaml
            ENV_YAML[floor] = f'maps/{floor}.yaml'
            name = f'{floor}_{rid}_{method}_k5'
            try:
                pgm, yml = stamp_pgm(
                    Path(ENV_YAML[floor]), placements, obj, name)
                # Pull start/goal from the named-routes catalog
                catalog = json.load(open(
                    f'results/northern_private/{floor}_named_routes.json'))
                route_meta = next((rt for rt in catalog['routes']
                                   if rt['route_id'] == rid), None)
                manifest.append({
                    'env': floor, 'route': rid, 'method': method, 'k': 5,
                    'pgm': pgm, 'yaml': yml,
                    'placements_grid': placements,
                    'start_world': route_meta['start_world'] if route_meta else None,
                    'goal_world': route_meta['goal_world'] if route_meta else None,
                    'source': 'nph_named',
                })
            except Exception as e:
                print(f'Failed: {name}: {e}')

    # Named paper routes (k=5 greedy)
    p3 = Path('results/additional_experiments/exp2b/placements.json')
    if p3.exists():
        for r in json.load(open(p3)):
            if r.get('error') or not r.get('placements'):
                continue
            placements = r['placements']
            obj = ENV_OBJ['hospital']
            name = f'hospital_{r["route"]}_greedy_k5'
            try:
                pgm, yml = stamp_pgm(
                    Path(ENV_YAML['hospital']), placements, obj, name)
                manifest.append({
                    'env': 'hospital', 'route': r['route'],
                    'method': 'greedy', 'k': 5,
                    'pgm': pgm, 'yaml': yml,
                    'placements_grid': placements,
                    'start_world': r.get('start_world'),
                    'goal_world': r.get('goal_world'),
                    'source': 'exp2b',
                })
            except Exception as e:
                print(f'Failed: {name}: {e}')

    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(f'Stamped {len(manifest)} maps. Manifest: {MANIFEST}')


if __name__ == '__main__':
    main()
