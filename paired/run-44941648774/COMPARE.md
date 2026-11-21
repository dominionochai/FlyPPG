| condition | arm | runs | median kOk % (IQR) | median chest-not-visible % (IQR) | pulse samples (IQR) | pulse confidence (IQR) | pulse rate SD (IQR) | breathing samples (IQR) | breathing confidence (IQR) | breathing rate SD (IQR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | control | 3 | 0.00 (48.97) | 0.00 (0.31) | 0.00 (64.50) | 62.26 (0.00) | 11.50 (0.00) | 0.00 (14.50) | 85.37 (0.00) | 0.71 (0.00) |
| static | fly_stab | 3 | 5.85 (4.48) | 0.00 (10.01) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |
| static | flyppg | 3 | 0.00 (0.01) | 0.00 (33.75) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |
| static | surround | 3 | 16.60 (28.79) | 27.07 (41.63) | 21.00 (51.00) | 50.53 (0.68) | 9.42 (2.18) | 0.00 (0.00) | n/a | n/a |
| stress | control | 3 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |
| stress | fly_stab | 3 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |
| stress | flyppg | 3 | 0.00 (2.85) | 0.00 (0.17) | 0.00 (5.50) | 57.89 (0.00) | 7.80 (0.00) | 0.00 (0.00) | n/a | n/a |
| stress | surround | 3 | 0.00 (40.22) | 0.00 (0.00) | 0.00 (88.00) | 94.30 (0.00) | 0.57 (0.00) | 0.00 (49.50) | 64.92 (0.00) | 2.91 (0.00) |

Three independent SDK invocations per arm and condition; execution order was shuffled with a fixed seed.
Control is decode/re-encode only. surround applies gated background-referenced illumination normalization only. fly_stab applies FlyPPG motion compensation only. flyppg combines illumination normalization and FlyPPG motion compensation.
The stress clip adds deterministic mild camera translation/rotation and brightness flicker before preprocessing.
These are SDK output/validation and repeatability measures only. This dataset run has no pulse ground-truth reference, so the table does not establish BPM accuracy.
