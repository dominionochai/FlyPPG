# Three-day integration and ship gate

## Decision

The SmartSpectra integration path is technically supported. Presage's Node.js SDK exposes both:

- `useFile(videoPath)` for prerecorded-video baseline runs.
- `useCustomInput()` + `sendFrame(...)` for application-supplied RGB frames.

The second path is the intended seam for FlyPPG preprocessing. The new
`experiments/smartspectra_custom_input_run.mjs` runner accepts a video that
has already been preprocessed and submits its decoded RGB frames to
SmartSpectra. It records the SDK measurements and validation events.

This runner is an integration adapter, not proof that the current FlyPPG
preprocessor improves measurements. It must first be exercised with a valid
SmartSpectra key and a known-good video.

## Current evidence and stop conditions

- The existing single-recording pilot reported raw POS MAE 20.15 BPM and
  FlyPPG v3 MAE 24.39 BPM. On that recording, v3 was worse by 4.24 BPM.
- SmartSpectra has not produced measurements because the hosted SDK requests
  received HTTP 403 responses.
- The pilot uses one recording and overlapping windows; it cannot support a
  general superiority claim.
- Do not claim improved accuracy, low-light robustness, motion robustness,
  skin-tone parity, or medical validity without the corresponding tests.

## Three-day execution gate

### Day 1 — unblock and establish the comparator

1. Presage resolves the authorization failure.
2. Run the unmodified source video through `useFile()`.
3. Confirm actual pulse samples, timestamps, confidence/stability fields where
   available, and coverage.
4. Save SDK version, input metadata, validation events and raw output.

### Day 2 — test the preprocessing interface

1. Run the same source through the current FlyPPG preprocessing candidate.
2. Submit that output through `smartspectra_custom_input_run.mjs`.
3. Include a re-encode-only control to detect codec/resampling effects.
4. Compare source-video SmartSpectra against preprocessed-video SmartSpectra
   on exactly aligned reference windows.
5. If preprocessing degrades the result or produces no reliable gain, do not
   ship it as an enhancement.

### Day 3 — decision and packaging

Ship only if the same-recording paired comparison shows a repeatable benefit
without unacceptable loss of coverage. Clearly label the result as a pilot.
Otherwise, package the working baseline and report the negative or inconclusive
result; do not manufacture a success claim.

## Hardware boundary

The low-cost FlyLight camera/illumination accessory remains a concept. The
current three-day deliverable is software-only. A 3D enclosure render may be
used to communicate a proposed design, but it must be labelled as a concept,
not functioning hardware. No custom camera hardware should be purchased or
manufactured before the software experiment demonstrates a benefit.

## Important integration limitation

The custom-input API accepts RGB frames. The preprocessing output must preserve
valid frame timing, dimensions and pixel format. This runner sends frames at
the source frame rate with monotonically increasing timestamps. Confirm the
SDK's accepted resolution and frame-rate constraints with the actual SDK run;
do not assume every video size is supported.
