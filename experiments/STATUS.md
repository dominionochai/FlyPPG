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


## Latest hosted execution attempt

**2026-11-19 — GitHub Actions run 44451617172**

- Python test workflow: passed on the same commit.
- UBFC real-video workflow: failed before video processing.
- The ground-truth text file downloaded successfully (74.3 kB).
- The official Google Drive video download was blocked by Google's "too many users have viewed or downloaded this file recently" quota response.
- The FlyPPG benchmark script was therefore skipped; no real-video metrics or output artifact exist.
- The separate SmartSpectra workflow has not been executed. It requires the video plus a configured `SMARTSPECTRA_API_KEY` repository secret.

Run: https://github.com/dominionochai/FlyPPG/actions/runs/44451617172

The official UBFC page also exposes a OneDrive link, but the available browser route redirects to Microsoft sign-in, so anonymous access has not been verified. Do not substitute an unlicensed third-party copy or claim the benchmark ran.
