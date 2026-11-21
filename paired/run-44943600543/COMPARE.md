| condition | arm | runs | median kOk % (IQR) | median chest-not-visible % (IQR) | pulse samples (IQR) | pulse confidence (IQR) | pulse rate SD (IQR) | breathing samples (IQR) | breathing confidence (IQR) | breathing rate SD (IQR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | control | 1 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | n/a | n/a | 0.00 (0.00) | n/a | n/a |

Three independent SDK invocations per arm and condition; execution order was shuffled with a fixed seed.
Control is decode/re-encode only. surround applies gated background-referenced illumination normalization only. fly_stab applies FlyPPG motion compensation only. flyppg combines illumination normalization and FlyPPG motion compensation.
The stress clip adds deterministic mild camera translation/rotation and brightness flicker before preprocessing.
These are SDK output/validation and repeatability measures only. This dataset run has no pulse ground-truth reference, so the table does not establish BPM accuracy.
