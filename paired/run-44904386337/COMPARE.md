| condition | arm | runs | median kOk % (IQR) | median chest-not-visible % (IQR) | pulse samples (IQR) | pulse confidence (IQR) | pulse rate SD (IQR) | breathing samples (IQR) | breathing confidence (IQR) | breathing rate SD (IQR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | control | 3 | 98.13 (26.08) | 0.54 (11.94) | 135.00 (59.00) | 59.40 (6.67) | 11.36 (1.66) | 37.00 (41.50) | 73.04 (4.12) | 2.69 (1.67) |
| static | fly_stab | 3 | 7.85 (1.79) | 0.00 (26.93) | 0.00 (11.50) | 40.96 (0.00) | 6.97 (0.00) | 0.00 (0.00) | n/a | n/a |
| static | flyppg | 3 | 0.03 (3.95) | 55.04 (33.97) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |
| static | surround | 3 | 40.05 (38.04) | 57.16 (38.26) | 51.00 (56.00) | 56.24 (16.36) | 11.58 (1.35) | 3.00 (21.00) | 78.37 (0.47) | 3.39 (1.92) |
| stress | control | 3 | 63.96 (7.42) | 0.00 (0.03) | 177.00 (7.00) | 88.04 (1.64) | 1.55 (0.32) | 99.00 (13.00) | 66.56 (0.36) | 2.82 (0.15) |
| stress | fly_stab | 3 | 26.20 (1.25) | 0.00 (0.02) | 56.00 (1.00) | 53.84 (2.90) | 9.45 (0.82) | 0.00 (0.00) | n/a | n/a |
| stress | flyppg | 3 | 0.03 (3.06) | 0.00 (0.00) | 0.00 (4.50) | 47.72 (0.00) | 3.83 (0.00) | 0.00 (0.00) | n/a | n/a |
| stress | surround | 3 | 70.97 (5.09) | 0.00 (10.26) | 169.00 (36.50) | 76.93 (1.64) | 6.03 (0.55) | 82.00 (19.50) | 64.84 (0.96) | 2.80 (0.87) |

Three independent SDK invocations per arm and condition; execution order was shuffled with a fixed seed.
Control is decode/re-encode only. surround applies gated background-referenced illumination normalization only. fly_stab applies FlyPPG motion compensation only. flyppg combines illumination normalization and FlyPPG motion compensation.
The stress clip adds deterministic mild camera translation/rotation and brightness flicker before preprocessing.
These are SDK output/validation and repeatability measures only. This dataset run has no pulse ground-truth reference, so the table does not establish BPM accuracy.
