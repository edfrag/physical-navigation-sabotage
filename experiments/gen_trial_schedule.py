"""Block-randomized trial schedule for hardware Phase 4.

Produces N_PER_COND trials per condition, randomized in blocks of
BLOCK_SIZE so every block contains one trial of each condition (when
BLOCK_SIZE = number of conditions). Protects against time-of-day
confounds (battery drift, Wi-Fi load).

Usage:
  python3 -m experiments.gen_trial_schedule \
      --conditions no_attack greedy_k1 greedy_k2 \
      --trials-per-condition 10 \
      --block-size 3 \
      --seed 42 \
      --out results/hw/trial_schedule.json
"""
import argparse
import json
import random
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='+', required=True)
    ap.add_argument('--trials-per-condition', type=int, default=10)
    ap.add_argument('--block-size', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_cond = len(args.conditions)
    if args.block_size != n_cond:
        print(f'WARN: block_size ({args.block_size}) != n_conditions ({n_cond}). '
              'Recommend block_size == n_conditions for clean rotation.')

    # Construct full ordered pool: each condition repeated N_PER_COND times
    pool = []
    for cond in args.conditions:
        pool.extend([cond] * args.trials_per_condition)

    # Block-randomize: shuffle within blocks of BLOCK_SIZE so each block
    # contains exactly one trial of each condition (if BLOCK_SIZE == n_cond)
    n_total = len(pool)
    if n_total % args.block_size != 0:
        print(f'WARN: total trials ({n_total}) not divisible by block_size ({args.block_size}).')

    blocks = []
    if args.block_size == n_cond:
        # Clean rotation: each block has one of each condition
        for b in range(args.trials_per_condition):
            block = list(args.conditions)
            rng.shuffle(block)
            blocks.append(block)
    else:
        # Generic shuffle
        rng.shuffle(pool)
        for i in range(0, n_total, args.block_size):
            blocks.append(pool[i:i + args.block_size])

    schedule = []
    for bi, block in enumerate(blocks):
        for ti, cond in enumerate(block):
            schedule.append({
                'trial': len(schedule) + 1,
                'block': bi + 1,
                'within_block': ti + 1,
                'condition': cond,
            })

    out = {
        'config': vars(args),
        'n_total_trials': len(schedule),
        'schedule': schedule,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f'Wrote {args.out} with {len(schedule)} trials in {len(blocks)} blocks')

    # Also a human-readable cheat sheet
    md = ['# Trial schedule', '',
          f'Total: {len(schedule)} trials, {len(blocks)} blocks of {args.block_size}',
          '', '| trial | block | condition |', '|---|---|---|']
    for t in schedule:
        md.append(f'| {t["trial"]:>3} | {t["block"]:>2} | {t["condition"]} |')
    Path(args.out).parent.joinpath('trial_schedule.md').write_text('\n'.join(md))


if __name__ == '__main__':
    main()
