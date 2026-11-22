| condition | reps | usable pulse samples | window coverage % | MAE vs clean (bpm) | at-flicker-freq % | frames passing gate % | validation events |
|---|---|---|---|---|---|---|---|
| clean (control) | 2 | 50 | 82 | 0.00 | 0 | 96 | 1801 |
| clean + nearref | 2 | 50 | 91 | 2.78 | 40 | 100 | 1801 |
| 6% flicker | 2 | 63 | 91 | 7.88 | 10 | 98 | 1801 |
| 6% flicker + denotch | 2 | 27 | 64 | 8.64 | 0 | 96 | 1801 |
| 6% flicker + nearref | 2 | 23 | 64 | 9.68 | 29 | 98 | 1801 |
| 12% flicker | 2 | 37 | 55 | 7.04 | 0 | 97 | 1801 |
| 12% flicker + denotch | 2 | 45 | 82 | 2.70 | 22 | 100 | 1801 |
| 12% flicker + nearref | 2 | 39 | 64 | 8.15 | 0 | 100 | 1801 |
| entrain 88bpm | 2 | 47 | 64 | 8.88 | 100 | 97 | 1801 |
| entrain 88bpm + denotch | 2 | 34 | 82 | 6.78 | 67 | 97 | 1801 |

Noise floor (clean repeat vs clean repeats, leave-one-out): 0.00 bpm. Pseudo-reference = SmartSpectra own clean readings; no ground-truth pulse.

**Verdicts**
- 06% flicker + denotch: SmartSpectra DEGRADED (MAE 7.88, coverage 91%); with FlyPPG MAE 8.64, coverage 64% -> WORSE (threshold 1.00 bpm / 15 pts)
- 06% flicker + nearref: SmartSpectra DEGRADED (MAE 7.88, coverage 91%); with FlyPPG MAE 9.68, coverage 64% -> WORSE (threshold 1.00 bpm / 15 pts)
- 12% flicker + denotch: SmartSpectra DEGRADED (MAE 7.04, coverage 55%); with FlyPPG MAE 2.70, coverage 82% -> IMPROVED (threshold 1.00 bpm / 15 pts)
- 12% flicker + nearref: SmartSpectra DEGRADED (MAE 7.04, coverage 55%); with FlyPPG MAE 8.15, coverage 64% -> WORSE (threshold 1.00 bpm / 15 pts)
- entrainment trap 88 bpm (raw): ENTRAINED (reading locked to flicker frequency) — 100% of windows at the flicker frequency, MAE vs clean 8.88
- entrainment trap 88 bpm (denotch): ENTRAINED (reading locked to flicker frequency) — 67% of windows at the flicker frequency, MAE vs clean 6.78