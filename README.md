# FlyPPG

**Biologically inspired video-preprocessing research for remote photoplethysmography (rPPG).**

FlyPPG investigates whether visual-processing principles from the fruit-fly optic lobe can make camera-based pulse measurement more resilient to illumination changes and motion. It is designed as an experimental preprocessing layer around Presage SmartSpectra—not as a replacement pulse estimator.

> **Current result: no demonstrated advantage over a strong conventional rPPG baseline.** The repository preserves the negative findings rather than presenting the earlier weak-baseline gains as a success. The SmartSpectra adapter and benchmark harness exist, but a valid end-to-end SmartSpectra comparison has not yet been completed.

## Why this exists

Camera-based rPPG extracts subtle blood-volume-related colour changes from video. Camera motion, auto-exposure, flicker, compression and changing illumination can interfere with those signals.

FlyPPG explores two biological ideas:

- **Surround-based gain control:** estimate shared illumination variation from a background region and apply a conservative correction only when foreground/background signals are strongly coupled.
- **Motion processing inspired by fly T4/T5 pathways:** use a compact, connectome-derived spatiotemporal feature bank to estimate local motion.

The biological inspiration is a design hypothesis, not evidence that the implementation reproduces a fly visual system. Several model parameters and mappings are engineering assumptions.

## What is implemented

- Video preprocessing with matched control, conventional, classical-flow, surround-normalisation, FlyPPG-stabilisation and combined arms.
- A connectome-derived T4/T5 feature bank and local motion readout.
- Gated background-referenced illumination correction.
- SmartSpectra file/custom-input runner experiments and a paired evaluation harness.
- A scorer that retains SDK confidence, stability and timestamps, reports coverage separately, and compares paired outputs without silently dropping failed arms.
- Synthetic stress tests and an interactive visualization of the modeled fly visual pathway.

## What the evidence says

The most informative result is the comparison against POS (Plane-Orthogonal-to-Skin), a standard chrominance rPPG baseline—not the earlier green-channel proxy.

In the repository's synthetic lighting tests:

- POS alone achieved low error across the tested lighting conditions.
- Background-referenced normalisation failed when face and background illumination were decoupled, or when the background contained distractors/clutter.
- Adding surround normalisation to POS produced at most small changes; the differences were too small to establish a robust improvement.
- In the synthetic motion benchmark, FlyPPG did not outperform strong conventional preprocessing. Under the hardest combined condition, all tested methods degraded and conventional smoothing was best.

The separate low-light/compression experiment also did not establish a benefit from adaptive pooling.

**Interpretation:** the current evidence does not support claiming better BPM accuracy, reliability, or motion robustness than conventional rPPG preprocessing. The detailed results and caveats are in [REPORT.md](REPORT.md).

## SmartSpectra integration and validation status

The repository includes code for preprocessing real video files and passing files/frames into SmartSpectra experiments. However, a recent hosted SDK attempt was blocked by exhausted SmartSpectra measurement credits and returned an invalid no-output status. That run is an infrastructure failure, not a valid algorithm comparison.

The UBFC single-recording workflow also currently crashes in its native processing step (double-free/corruption); its output is not treated as benchmark evidence.

No claim of physiological accuracy is made from SDK frame-validation events or sample counts. A valid accuracy study requires synchronized video and ECG/PPG ground truth, subject-disjoint evaluation, repeated runs, and reporting of both error and coverage.

See:
- [SmartSpectra benchmark protocol](SS_BENCH.md)
- [Engineering audit and claim boundaries](AUDIT.md)
- [Current experiment status](experiments/STATUS.md)
- [Submission summary](SUBMISSION.md)

## Run the tests

Python 3.11+ is recommended.

```bash
python -m pip install numpy scipy opencv-python-headless
python -m unittest discover -s tests -v
```

The tests cover preprocessing gates, signal-region handling, benchmark parsing and status reporting. Passing unit tests verifies those code paths; it does not establish physiological accuracy or SDK performance.

For the SmartSpectra experiment setup, see [SS_BENCH.md](SS_BENCH.md). Keep API keys in environment variables; never commit them.

## Repository map

| Path | Purpose |
|---|---|
| flyppg3.py, fly_conn.py | Synthetic pipeline and connectome-inspired motion features |
| ss_preprocess.py, stabilize.py | Video preprocessing arms |
| experiments/ | SmartSpectra runners, benchmark scripts and diagnostics |
| ss_score.py | SmartSpectra output parsing and paired scoring |
| tests/ | Regression and parser tests |
| REPORT.md, AUDIT.md | Results, limitations and engineering audit |
| brain3d.html | Interactive visualization |

## Safety and scope

FlyPPG is research software, not a medical device. It must not be used for diagnosis, treatment, or medical decisions. SmartSpectra's measurements are wellness/informational outputs and are not a substitute for clinical assessment.
