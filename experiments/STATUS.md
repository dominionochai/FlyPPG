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


## LGI-PPGI access probe — 2026-11-19

- Evaluated the LGI-PPGI dataset as an alternative. Its repository describes 100 webcam recordings from 25 participants with synchronized finger pulse-oximeter reference data and lists the dataset under CC BY 4.0.
- The published archive endpoint was probed from GitHub Actions using HEAD and a 1 KiB HTTP byte-range request. Both requests timed out after 25 seconds; the server returned no HTTP status (curl status 000).
- No archive bytes were retained and no video was downloaded. Range access and partial-archive extraction are not available to us through this endpoint at present.
- This is an access failure, not evidence that the dataset itself is invalid or that its licence is unsuitable. Do not claim an LGI real-video benchmark has run.

## Next decision

Stop spending cycles retrying the inaccessible UBFC Google Drive link and LGI archive endpoint. The benchmark harness and tests are ready, but real-data validation is blocked on obtaining one authorized, downloadable face-video recording with its synchronized reference signal. Do not use unlicensed mirrors, research-only datasets for commercial claims, or synthetic data as a substitute for real-video evidence. Once an authorized recording is available, run FlyPPG and POS on the same recording first; add SmartSpectra only when its licensed runtime and credentials are configured.


## MCD-rPPG single-recording access check — 2026-11-19

**Hosted run:** https://github.com/dominionochai/FlyPPG/actions/runs/44453665031  
**Pinned dataset revision:** `akramic/mcd_rppg@04df5b6`  
**Recording:** `1020_IriunWebcam_before`

- Downloaded the video (16,252,928 bytes) and matching `ppg_sync` file (51,219 bytes) successfully in GitHub Actions. No dataset files were committed to this repository or retained as workflow artifacts.
- FFprobe identified an MPEG-4 video stream at 640×480 and nominal 24 fps; it counted 4,310 frames. FFmpeg decoded the first frame, but emitted a macroblock decode warning during inspection.
- The paired reference contains 4,310 rows and two numeric columns. The second column's intervals have a median of 0.003953 seconds and sum to 15.731 seconds.
- **This is not yet a validated synchronized benchmark input.** The video reports a zero duration in its container metadata, and its nominal frame count/rate imply about 179.6 seconds, while the reference intervals sum to about 15.7 seconds. The timestamp/sample semantics and video timing must be resolved before aligning signals.
- No FlyPPG or POS score was calculated. Do not report this as a real-data accuracy result.

The access gate is now passed for this one recording. The next engineering task is to resolve the timing discrepancy and verify full-frame decoding before any model comparison.


## MCD timing diagnosis — 2026-11-19

The earlier apparent 15.7-second reference duration was a parser mistake: it summed the second reference column as if every value were a sample interval. That interpretation is unsupported and must not be used.

A corrected hosted inspection on commit `d4d0feb` found:

- 4,310 decoded video-frame timestamps, from 0.000000 to 179.541667 seconds.
- 4,310 reference rows: exactly one row per decoded video frame.
- Reference column 1 spans 0–255 and is consistent with the PPG waveform values.
- Reference column 2 spans 0–0.008977 and is not monotonic; summing it does not produce recording duration. Its exact meaning still requires confirmation from the dataset's official preprocessing code.
- The AVI reports a zero container duration despite valid per-frame timestamps.
- Full decode emits MPEG-4 macroblock errors (including around macroblock 108). The workflow exits successfully, but this warning means video integrity is not yet cleared.

**Conclusion:** the 15.7-second discrepancy is not evidence of a 15.7-second reference against a 179.5-second video. Row counts and per-frame timestamps now align structurally. Do not score this recording yet: identify column 2 semantics and quantify damaged frames first.

**Corrected hosted run:** https://github.com/dominionochai/FlyPPG/actions/runs/44454005357


## Reference-file semantics check — 2026-11-19

Inspected the published MCD-rPPG baseline documentation and its dataset issue tracker. The baseline repository identifies `ppg_sync/` as the synchronized PPG signal associated with each video. A separate published dataset analysis explicitly identifies **column 1** of `ppg_sync/*.txt` as the synchronized PPG ground truth and warns against using the discrete clinical `pulse` biomarker as video-window ground truth.

For this recording, column 1 therefore contains the reference waveform (values 0–255), not timestamps or heart-rate values. The 4,310-row correspondence with the 4,310 video frames is consistent with the per-frame synchronized reference representation used by this recording. Column 2 remains uninterpreted; do not sum it or use it as elapsed time without confirming its definition in the official data-generation code.

Sources:
- https://github.com/garrulus2003/MCD-rPPG-baselines
- https://github.com/ksyegorov/mcd_rppg/issues/5
- https://github.com/ksyegorov/mcd_rppg/issues/8

**Benchmark gate remains closed:** quantify and isolate MPEG-4 decode corruption, then validate the waveform-to-video sampling relationship before calculating HR error.


## MCD same-recording baseline — 2026-11-19

**Hosted run:** https://github.com/dominionochai/FlyPPG/actions/runs/44455239442  
**Artifact:** `mcd-flyppg-smartspectra-baseline`  
**Recording:** `1020_IriunWebcam_before.avi`, 4,310 frames, 24 fps, 179.58 s  
**Reference:** synchronized `ppg_sync` column 1, scored in 10-second windows with 5-second stride.

| Arm | Windows | Coverage | MAE (BPM) | RMSE (BPM) | Within 5 BPM |
|---|---:|---:|---:|---:|---:|
| Raw POS | 34/34 | 100% | 20.15 | 24.56 | 11.76% |
| FlyPPG v3 stabilization + POS | 34/34 | 100% | 24.39 | 30.18 | 8.82% |
| SmartSpectra | 0/34 | 0% | Not measured | Not measured | 0% |

**Measured result:** On this recording and this pipeline, FlyPPG v3 is 4.24 BPM worse in MAE than raw POS. This is a negative preliminary result, not a claim about general performance. The run used the centered ROI fallback because the hosted OpenCV build did not expose `CascadeClassifier`; the recording also has a corrupt final AVI packet. Reference HR was estimated from the synchronized PPG waveform's dominant 0.7–3.0 Hz frequency in each window.

**SmartSpectra blocker:** the workflow environment had an empty `SMARTSPECTRA_API_KEY`; the SDK runner stopped before processing. Therefore there is still no measured FlyPPG-vs-SmartSpectra result. The exact same-video workflow is now implemented and retains the FlyPPG baseline artifact even when the SDK cannot run. Configure the repository Actions secret `SMARTSPECTRA_API_KEY` with the licensed key, then rerun `MCD FlyPPG vs SmartSpectra baseline`. Do not report SmartSpectra as zero BPM or treat the missing run as a score.
