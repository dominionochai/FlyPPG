# FlyPPG engineering audit

Date: 2026-11-19
Branch: `benchmark/integrity-audit`
Scope: repository inspection and SmartSpectra public C++ sample/docs inspection. No real SmartSpectra run was performed.

## Executive finding

FlyPPG has not demonstrated superiority over SmartSpectra. The repository's own `REPORT.md` says its synthetic POS comparisons show no FlyPPG advantage. The immediate engineering priority is a trustworthy evaluation path, not another preprocessing algorithm.

## Verified repository facts

- `README.md` and `REPORT.md` explicitly label existing results as synthetic/proxy-extractor results.
- The latest report says POS performs strongly across the simulated lighting cases and the FlyPPG surround-normalisation claim does not hold against POS.
- The connectome-derived motion readout is fitted using synthetic videos and synthetic displacement targets.
- `SS_BENCH.md` describes a head-to-head protocol, but the real SDK output schema and reference alignment were not validated by a real run.
- `ss_score.py` currently intersects videos across every arm before computing each arm's MAE/RMSE. This means a missing result in any one arm removes that video from every arm's accuracy calculation. Coverage is calculated separately, so the reported accuracy and coverage do not use the same population.
- The current reference CSV has one constant `hr_bpm` per video. That is inadequate for recordings with material within-video heart-rate changes, such as stress tasks.
- The current scorer takes the median of the later half of the returned pulse-rate values. It does not use confidence, stability, or sample timestamps.

## SmartSpectra SDK contract checked against its public source

Inspected:
- `Presage-Security/SmartSpectra/cpp/samples/minimal_example/main.cc`
- `Presage-Security/SmartSpectra/cpp/samples/minimal_example/README.md`
- `Presage-Security/SmartSpectra/cpp/docs/metrics.md`
- `Presage-Security/SmartSpectra/cpp/docs/device-readiness.md`

Verified:
- The minimal example emits `Got edge metrics at <timestamp> microseconds: <protobuf JSON>`.
- Pulse rate is nested under `cardio.pulseRate` in protobuf JSON; the measurement has `value`, `confidence`, `timestamp`, and `stable` fields.
- The existing parser's lowercasing means it can recognize `pulseRate` as `pulserate`; it was not correct to claim that camelCase alone breaks the existing parser.
- The existing parser does discard confidence/stability/timestamp metadata.
- The SDK docs say file `SetInterframeDelay` changes playback speed, not measured frame rate. Do not treat the default delay as a sample-rate conversion; record it because it changes experiment runtime.
- The SDK docs state the metrics are for wellness/informational purposes and are not cleared for diagnosis or treatment.

## Changes on this branch

1. Reworked `ss_score.py`:
   - Parses both lowerCamelCase and snake_case pulse-rate fields.
   - Retains confidence, stable flag and timestamp metadata.
   - Calculates each arm's accuracy on that arm's own available outputs.
   - Calculates each paired comparison on the videos available in that arm and the control, rather than intersecting every arm.
   - Bootstraps paired differences by video, not by overlapping windows.
   - Reports coverage and the proportion within 5 BPM using all reference videos as denominator.
   - Emits a JSON report that explicitly warns that one constant reference HR per video cannot represent time-varying physiology.
2. Added `tests/test_ss_score.py` for protobuf field parsing, string-encoded int64 timestamps, malformed lines, filters and pairwise comparisons.
3. Rewrote `SS_BENCH.md` to state the limitations and interpretation rules.

## Not yet verified

- The new scorer tests have been added but have not been executed in this environment.
- No actual SmartSpectra SDK binary or API key was used.
- No real video/reference dataset was processed.
- No FlyPPG algorithm improvement is claimed.
- No reference-signal/video timestamp alignment has been implemented yet.

## Required next engineering steps

1. Run `python -m unittest discover -s tests -v` and fix any failures.
2. Run one short, legally usable real recording through the unmodified SmartSpectra SDK. Confirm pulse values, timestamps, confidence, stability, errors and coverage.
3. Validate video/reference synchronization and derive reference HR per SDK-compatible time window. Do not use one mean HR for an entire stress recording.
4. Freeze a subject-disjoint evaluation split before tuning.
5. Compare SmartSpectra control, POS, and each preprocessing arm on identical clips. Preserve failures in coverage and report paired video-level confidence intervals.
6. Only then decide whether a new FlyPPG algorithm is justified.

## Claim policy

Until the real head-to-head evaluation is complete, permitted wording is: "FlyPPG is an experimental rPPG preprocessing research prototype." Do not claim it is more accurate, more stable, or more reliable than SmartSpectra.


## Automated test workflow

GitHub Actions workflow `.github/workflows/tests.yml` was added to `main`. It compiles Python sources and runs unittest discovery on pull requests and pushes to main. The audit branch is awaiting its first CI run.
