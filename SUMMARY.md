# Reported headline results

This file records values reported in the camera-ready manuscript. It is not a substitute for the original raw result files.

## Planner layer

- Hospital A named routes: greedy AOPP denied 3/4 routes under Nav2 NavFn.
- Hospital A 30-route sweep: the binary surrogate predicted 30/30 denials; NavFn confirmed 8/29 valid routes, approximately 3.7x over-prediction.
- Hospital B named routes: greedy denied 2/8 routes under NavFn.
- Hospital B validation contingency over 32 route/attacker pairs: TP=1, TN=25, FP=3, FN=3 (81.3% agreement).

## Gazebo

- Greedy k=3: 10/30 failures (33.3%).
- Greedy k=4: 5/20 failures (25.0%).
- Random k=3: 0/29 valid failures; one infrastructure error excluded.
- Fisher exact, greedy k=3 vs random k=3: p = 7.97e-4.
- MPPI qualitative transfer check: 5/5 failures on the same greedy k=3 placement.

## Hardware

HW-1, 40 trials per condition:

- No attack: 0/40 denial.
- Greedy k=1: 32/40 denial (80%).
- Greedy k=2: 40/40 denial (100%).
- Fisher exact: p = 7.02e-15 (no attack vs k=1); p = 1.86e-23 (no attack vs k=2).

HW-2, 10 trials per condition at k=3:

- Random: 1/10 failures.
- Greedy: 6/10 failures (two-sided Fisher p = 0.057).
- Median traversal time: no attack 118 s, random 132 s, greedy 282 s.
- Median recovery count: 0, 0, and 5 respectively.
