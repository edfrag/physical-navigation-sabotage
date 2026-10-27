"""Re-run NavFn on entries that failed with infrastructure errors.

Reads an existing navfn_*.json output, identifies entries with
infrastructure errors (action server unavailable, goal rejected, no
result, exception), and re-runs only those entries.

Usage (inside aopp-turtlebot4 container):
  python3 /workspace/scripts/nav2_retry_failures.py \
      --in /workspace/results/additional_experiments/route_sweep/navfn_30routes.json \
      --source route_sweep
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/workspace/scripts')
sys.stdout.reconfigure(line_buffering=True)

from nav2_verify_nph_named import run_navfn, kill_nav2

INFRA_ERRORS = {'action server unavailable', 'goal rejected', 'no result',
                'exception'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--source', required=True)
    args = ap.parse_args()

    project = Path('/workspace')
    manifest = json.load(open(
        project / 'results' / 'topology_baselines' / 'navfn_manifest.json'))
    src_entries = {(e['env'], e['route'], e['method']): e
                    for e in manifest if e.get('source') == args.source}

    results = json.load(open(args.in_path))
    failures = [r for r in results if not r.get('navfn_success')
                and any(e in (r.get('notes') or '') for e in INFRA_ERRORS)]
    print(f'Retrying {len(failures)} infra-error entries...')

    for r in failures:
        key = (r['env'], r['route'], r['method'])
        e = src_entries.get(key)
        if e is None:
            print(f'  {key}: not in manifest, skip')
            continue
        sw = e.get('start_world'); gw = e.get('goal_world')
        map_yaml = project / e['yaml']
        if not sw or not gw or not map_yaml.exists():
            print(f'  {key}: missing data, skip')
            continue
        print(f'  Retry: {key}')
        ok, plen, notes = run_navfn(map_yaml, sw, gw, timeout=45)
        # Update in-place
        r['navfn_success'] = ok
        r['navfn_path_length_m'] = plen
        r['notes'] = notes
        tag = f'PATH {plen:.1f}m' if ok else 'DENIAL'
        print(f'    -> {tag}  {notes}')
        with open(args.in_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print('\nDone.')


if __name__ == '__main__':
    main()
