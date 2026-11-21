| condition | arm | runs | median kOk % (IQR) | median chest-not-visible % (IQR) | pulse samples (IQR) | pulse confidence (IQR) | pulse rate SD (IQR) | breathing samples (IQR) | breathing confidence (IQR) | breathing rate SD (IQR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | control | 3 | 44.92 (30.52) | 31.82 (29.76) | 46.00 (44.50) | 47.49 (9.61) | 9.72 (2.52) | 21.00 (25.00) | 77.96 (0.71) | 1.87 (1.14) |
| static | fly_stab | 3 | 0.03 (5.36) | 0.00 (20.10) | 0.00 (10.50) | 44.68 (0.00) | 6.28 (0.00) | 0.00 (0.00) | n/a | n/a |
| static | flyppg | 3 | 7.60 (5.76) | 52.18 (33.80) | 0.00 (12.50) | 35.92 (0.00) | 5.48 (0.00) | 0.00 (0.00) | n/a | n/a |
| static | surround | 3 | 96.21 (0.44) | 2.56 (0.36) | 138.00 (6.00) | 55.58 (3.87) | 11.15 (0.10) | 80.00 (33.50) | 78.98 (1.63) | 2.93 (0.37) |
| stress | control | 3 | 66.34 (4.31) | 0.00 (0.78) | 176.00 (1.00) | 94.86 (4.98) | 0.52 (0.85) | 99.00 (4.50) | 68.06 (1.64) | 2.52 (0.12) |
| stress | fly_stab | 3 | 25.01 (5.72) | 0.00 (0.07) | 51.00 (27.50) | 55.17 (5.50) | 10.00 (0.91) | 0.00 (0.00) | n/a | n/a |
| stress | flyppg | 3 | 6.23 (0.75) | 0.00 (0.00) | 9.00 (5.50) | 41.98 (5.47) | 4.03 (0.75) | 0.00 (0.00) | n/a | n/a |
| stress | surround | 3 | 68.25 (5.39) | 0.00 (11.18) | 179.00 (54.00) | 72.12 (1.96) | 5.41 (0.78) | 100.00 (7.50) | 66.63 (1.61) | 2.98 (0.39) |

Three independent SDK invocations per arm and condition; execution order was shuffled with a fixed seed.
Control is decode/re-encode only. surround applies gated background-referenced illumination normalization only. fly_stab applies FlyPPG motion compensation only. flyppg combines illumination normalization and FlyPPG motion compensation.
The stress clip adds deterministic mild camera translation/rotation and brightness flicker before preprocessing.
These are SDK output/validation and repeatability measures only. This dataset run has no pulse ground-truth reference, so the table does not establish BPM accuracy.
