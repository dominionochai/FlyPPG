# FlyPPG — project summary

## One-line description

FlyPPG is an experimental, biologically inspired preprocessing layer for camera-based remote photoplethysmography (rPPG), exploring whether fruit-fly visual-circuit principles can improve video quality before measurement by Presage SmartSpectra.

## Problem

rPPG depends on subtle colour variation in video. Illumination changes, camera movement, exposure adjustment and compression can alter those signals. Preprocessing can help, but aggressive correction can also remove useful pulse information or distort the region SmartSpectra needs to observe.

## Approach

FlyPPG investigates two mechanisms inspired by insect vision:

1. **Gated surround normalisation.** A background reference is used only when its temporal illumination signal is sufficiently strong and correlated with the foreground. The correction is deliberately conservative.
2. **Adaptive spatial illumination compensation (experimental).** A bounded, achromatic local gain map targets uneven face illumination while limiting shadow amplification. It is implemented as separate `spatial` and `combined_lighting` preprocessing arms; its rPPG benefit has not yet been benchmarked.
3. **T4/T5-inspired local motion estimation.** A compact spatiotemporal feature bank, derived from published fly visual-circuit connectivity as an engineering approximation, estimates local motion fields.

The project includes matched preprocessing arms, SmartSpectra file/custom-input experiment runners, output parsing, scoring utilities, regression tests and an interactive visualization. The new adaptive spatial arm is documented in [ADAPTIVE_ILLUMINATION.md](experiments/ADAPTIVE_ILLUMINATION.md).

## What was learned

The strongest result so far is negative and important: early gains against a green-channel proxy did not survive comparison with POS, a standard chrominance rPPG method. POS already performed strongly in the synthetic lighting benchmark. Surround normalisation was fragile when background and foreground illumination differed. The FlyPPG motion path did not beat strong conventional motion preprocessing in the synthetic motion tests. Adaptive pooling did not establish a low-light advantage.

The current implementation therefore does **not** claim better pulse-rate accuracy or robustness than SmartSpectra or conventional preprocessing.

## Integration and validation status

A SmartSpectra integration/benchmark harness is present in the repository. The latest hosted SmartSpectra attempt could not produce measurements because the account's measurement credits were exhausted (INVALID_NO_OUTPUT). This is not a valid result for or against the algorithm. A separate UBFC workflow currently aborts in native processing and is also excluded from performance claims.

A proper next validation requires:
- restored SmartSpectra measurement access;
- synchronized video and ECG/PPG reference signals;
- subject-disjoint test data;
- repeated matched runs across control and preprocessing arms;
- reporting of accuracy, coverage, confidence and failures together.

## Why publish it

FlyPPG contributes an inspectable research prototype, a reproducible experiment structure, a biological design hypothesis, and documented negative results. It demonstrates why preprocessing ideas must be tested against strong baselines and why missing SDK output must never be mistaken for model performance.

## Run tests

```bash
python -m pip install numpy scipy opencv-python-headless
python -m unittest discover -s tests -v
```

See [README](README.md) for the repository map and [REPORT](REPORT.md) for detailed benchmark results.

## Limitations

- Synthetic benchmark results are not real-world physiological validation.
- The motion readout is trained on synthetic data.
- The connectome mapping is approximate; it is not a faithful reconstruction of biological computation.
- SmartSpectra performance superiority has not been demonstrated.
- This is research software, not a medical device.
