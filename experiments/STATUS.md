# FlyPPG real-face experiment status

**Report date:** 2026-11-19  
**Status:** experiment harness added; real-video run not yet executed.

## What is in place

- Added `experiments/ubfc_single.py`: loads one UBFC-rPPG face video and its synchronized reference file, extracts a consistent face crop, applies POS to raw frames and (when the synthetic-trained readout weights are present) two FlyPPG preprocessing arms, and writes per-window estimates plus a JSON summary.
- Added `experiments/README.md`: data layout, run command, privacy constraints and interpretation limits.
- Existing `SS_BENCH.md` and `ss_score.py` document the SmartSpectra SDK pipeline and correct per-arm coverage/pairing logic.
- Dataset selection: UBFC-rPPG is the first target because it contains short 30-fps webcam face videos and synchronized finger-oximeter reference signals. Start with one recording; do not download the full dataset.

## Evidence currently available

| Evidence | Status |
|---|---|
| Synthetic FlyPPG experiments | Existing results; POS remains a stronger baseline in reported simulations |
| Real face-video processing | Not run |
| Synchronized reference alignment | Implemented in the new single-recording script; not validated against a real file |
| POS vs FlyPPG on real faces | Not measured |
| SmartSpectra vs FlyPPG on real faces | Not measured; SDK execution still requires its licensed runtime and credentials |
| CI for current PR head | No workflow run or status check returned by the GitHub integration at report time |

## Immediate run requirements

1. One authorized UBFC-rPPG subject folder with `vid.avi` and `ground_truth.txt`.
2. Python dependencies: NumPy, SciPy and OpenCV.
3. `readout_W3.npy` for the FlyPPG-v3 stabilisation arm.
4. SmartSpectra SDK runtime and valid credentials for the separate SDK comparison.

## Important caveats

- The first script is a pipeline smoke test, not a publishable benchmark.
- Its Haar face crop is a basic ROI method and needs visual/detection-quality auditing.
- One subject cannot support a superiority claim. Results must be repeated across held-out subjects.
- The current window-level script compares POS after preprocessing. It does not itself run SmartSpectra.
- No numerical real-world result is reported because no real dataset file has been processed. Synthetic metrics must not be relabelled as face-video results.

## Decision

Do not claim FlyPPG beats POS or SmartSpectra. The next valid milestone is a successful single-recording run with a populated `windows.csv` and `report.json`, followed by an SDK run on the exact same source recording.
