| condition | reps | usable pulse samples | window coverage % | MAE vs clean (bpm) | frames passing gate % | validation events |
|---|---|---|---|---|---|---|
| clean (control) | 2 | 50 | 82 | 0.00 | 96 | 1801 |
| clean + FlyPPG | 2 | 42 | 64 | 4.87 | 100 | 1801 |
| 6% flicker | 2 | 63 | 91 | 7.88 | 98 | 1801 |
| 6% flicker + FlyPPG | 2 | 40 | 82 | 6.73 | 96 | 1801 |
| 12% flicker | 2 | 37 | 55 | 5.44 | 97 | 1801 |
| 12% flicker + FlyPPG | 2 | 51 | 73 | 6.34 | 95 | 1801 |

Noise floor (clean repeat vs clean repeats, leave-one-out): 0.00 bpm. Pseudo-reference = SmartSpectra own clean readings; no ground-truth pulse.

**Verdicts**
- 06% flicker: SmartSpectra DEGRADED (MAE 7.88, coverage 91%); with FlyPPG MAE 6.73, coverage 82% -> IMPROVED (threshold 1.00 bpm / 15 pts)
- 12% flicker: SmartSpectra DEGRADED (MAE 5.44, coverage 55%); with FlyPPG MAE 6.34, coverage 73% -> IMPROVED (threshold 1.00 bpm / 15 pts)