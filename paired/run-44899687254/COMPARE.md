| condition | arm | runs | median kOk % (IQR) | median chest-not-visible % (IQR) | pulse samples (IQR) | pulse confidence (IQR) | pulse rate SD (IQR) | breathing samples (IQR) | breathing confidence (IQR) | breathing rate SD (IQR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | control | 3 | 96.49 (34.76) | 2.23 (24.50) | 137.00 (61.50) | 58.10 (7.66) | 11.33 (1.64) | 78.00 (46.00) | 72.20 (20.11) | 2.64 (1.15) |
| static | fly_stab | 3 | 13.13 (10.48) | 0.00 (12.15) | 5.00 (27.50) | 53.82 (18.17) | 8.59 (5.33) | 0.00 (0.00) | n/a | n/a |
| static | flyppg | 3 | 6.69 (13.56) | 0.00 (0.00) | 0.00 (15.50) | 34.03 (0.00) | 10.38 (0.00) | 0.00 (0.00) | n/a | n/a |
| stress | control | 3 | 72.10 (1.79) | 0.43 (0.75) | 178.00 (8.00) | 86.88 (5.20) | 2.10 (1.63) | 118.00 (10.00) | 63.52 (4.02) | 2.85 (0.18) |
| stress | fly_stab | 3 | 24.10 (6.34) | 63.59 (37.03) | 57.00 (14.50) | 50.98 (2.43) | 9.20 (0.76) | 0.00 (0.00) | n/a | n/a |
| stress | flyppg | 3 | 5.53 (1.02) | 66.87 (14.36) | 12.00 (1.50) | 54.75 (6.43) | 3.96 (0.85) | 0.00 (0.00) | n/a | n/a |

Three independent SDK invocations per arm and condition; execution order was shuffled with a fixed seed.
Control is decode/re-encode only. fly_stab applies FlyPPG motion compensation. flyppg applies illumination normalization plus FlyPPG motion compensation.
The stress clip adds deterministic mild camera translation/rotation and brightness flicker before preprocessing.
These are SDK output/validation and repeatability measures only. This dataset run has no pulse ground-truth reference, so the table does not establish BPM accuracy.
