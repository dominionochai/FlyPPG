# Three-day integration and ship gate

## Objective

Deliver a reproducible software comparison: SmartSpectra on original footage
versus SmartSpectra receiving FlyPPG-preprocessed frames. Do not build custom
hardware during this validation window.

## Confirmed integration interface

Presage's Node.js SDK documents `useCustomInput()` and
`sendFrame(buffer, width, height, stride, pixelFormat, timestampUs)`.
This lets a host application supply RGB frames to SmartSpectra. The custom-input
runner exercises that documented interface.

An available integration route does not prove that FlyPPG improves results.
Only the paired experiment can establish that.

## Existing evidence

- Raw POS MAE: 20.15 BPM on one MCD recording.
- FlyPPG v3 MAE: 24.39 BPM on that recording.
- SmartSpectra has not produced measurements; hosted SDK requests returned
  HTTP 403.
- The 34 windows overlap and come from one recording. They are not independent
  population evidence.

## Three-day gate

### Day 1: establish SmartSpectra baseline
- Resolve Presage authorization.
- Run the original recording through `useFile()`.
- Save timestamped pulse outputs, validation events, SDK version and coverage.
- Verify alignment against synchronized reference PPG.

### Day 2: test preprocessing
- Process the same source through FlyPPG.
- Submit the resulting RGB frames with the custom-input runner.
- Include an unmodified decode/re-encode control.
- Compare original, control and FlyPPG-preprocessed inputs on identical
  reference-aligned intervals.
- Reject gains caused by changed frame rate, timestamps, resolution or dropped
  frames.

### Day 3: ship decision
Ship an enhancement claim only if the paired comparison shows repeatable
improvement without unacceptable loss of coverage. Otherwise report the result
as negative or inconclusive.

## Claims boundary

Do not claim superior accuracy, motion robustness, low-light performance,
skin-tone parity, medical validity or functioning hardware without the
corresponding tests. FlyLight remains a hardware concept.
