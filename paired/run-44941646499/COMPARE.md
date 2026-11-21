| condition | arm | runs | median kOk % (IQR) | median chest-not-visible % (IQR) | pulse samples (IQR) | pulse confidence (IQR) | pulse rate SD (IQR) | breathing samples (IQR) | breathing confidence (IQR) | breathing rate SD (IQR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | control | 3 | 0.00 (48.21) | 0.00 (1.23) | 0.00 (69.50) | 56.84 (0.00) | 10.71 (0.00) | 0.00 (59.50) | 79.84 (0.00) | 2.61 (0.00) |
| static | fly_stab | 3 | 7.89 (5.11) | 0.00 (23.53) | 0.00 (0.50) | 86.59 (0.00) | 0.00 (0.00) | 0.00 (0.00) | n/a | n/a |
| static | flyppg | 3 | 0.00 (0.01) | 0.00 (31.69) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |
| static | surround | 3 | 96.30 (49.33) | 0.00 (0.12) | 154.00 (82.00) | 57.44 (0.32) | 10.81 (0.56) | 106.00 (60.50) | 79.21 (2.07) | 2.63 (0.23) |
| stress | control | 3 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |
| stress | fly_stab | 3 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |
| stress | flyppg | 3 | 0.00 (11.52) | 0.00 (0.20) | 0.00 (23.00) | 34.94 (0.00) | 11.49 (0.00) | 0.00 (0.00) | n/a | n/a |
| stress | surround | 3 | 0.00 (34.73) | 0.00 (0.00) | 0.00 (85.00) | 89.68 (0.00) | 5.94 (0.00) | 0.00 (45.00) | 67.35 (0.00) | 2.40 (0.00) |

Three independent SDK invocations per arm and condition; execution order was shuffled with a fixed seed.
Control is decode/re-encode only. surround applies gated background-referenced illumination normalization only. fly_stab applies FlyPPG motion compensation only. flyppg combines illumination normalization and FlyPPG motion compensation.
The stress clip adds deterministic mild camera translation/rotation and brightness flicker before preprocessing.
These are SDK output/validation and repeatability measures only. This dataset run has no pulse ground-truth reference, so the table does not establish BPM accuracy.
