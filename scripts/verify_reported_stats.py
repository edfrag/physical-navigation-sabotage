#!/usr/bin/env python3
"""Recompute statistical values that are recoverable from manuscript counts."""
from scipy.stats import fisher_exact


def fisher(label, a, b):
    # a=(success, failure), b=(success, failure)
    table = [[a[1], a[0]], [b[1], b[0]]]
    _, p = fisher_exact(table, alternative="two-sided")
    print(f"{label}: p={p:.8g}")


fisher("Gazebo greedy k=3 vs random k=3", (20, 10), (29, 0))
fisher("HW-1 no attack vs greedy k=1", (40, 0), (8, 32))
fisher("HW-1 no attack vs greedy k=2", (40, 0), (0, 40))
fisher("HW-2 random k=3 vs greedy k=3", (9, 1), (4, 6))

agreement = (1 + 25) / 32
print(f"Hospital B surrogate/NavFn agreement: {agreement * 100:.1f}%")
print(f"Hospital A surrogate over-prediction ratio: {30 / 8:.3f}x (paper rounds/describes as approximately 3.7x)")
