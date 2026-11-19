# SmartSpectra head-to-head harness

This harness measures whether video preprocessing changes SmartSpectra pulse-rate output on identical source recordings. It does not establish medical validity or superiority until the SDK is run against independently referenced real recordings.

## Protocol

- Keep a re-encode-only control arm.
- Run every arm on the same source videos and preserve source frame rate, dimensions, duration and timestamps.
- Do not tune parameters on the evaluation recordings.
- Record SDK version, OS, hardware, input FPS, resolution, codec, and all SDK warnings/errors.
- Primary comparison: per-video MAE difference against control, paired bootstrap resampled by video.
- Report coverage independently for every arm. A missing pulse estimate is a failed output, not a row to silently discard.
- Report RMSE and fraction of all reference videos within 5 BPM alongside MAE.
- Keep subject-level separation when multiple recordings come from the same participant.

## Reference CSV

Current scorer accepts one approximately stationary reference heart rate per video:

```csv
video,hr_bpm
subject01_task1,74.2
subject02_task1,81.0
```

Do not use a single mean HR for a long recording if the participant's pulse changes materially. The scorer explicitly labels this limitation in its JSON report. The next evaluation extension should use timestamp-aligned reference windows from synchronized BVP/ECG.

## Run

```powershell
pip install numpy opencv-python-headless scipy imageio-ffmpeg
python ss_preprocess.py --in videos --out processed
$env:SMARTSPECTRA_API_KEY = "<your key>"
python ss_run.py --exe .\build\samples\minimal_example\minimal_example.exe --dir processed --out ss_out
python ss_score.py --out ss_out --ref ref.csv --json score.json
Remove-Item Env:SMARTSPECTRA_API_KEY
```

The API key must remain in an environment variable and must never be committed or printed.

## SDK output contract

The C++ sample emits lines shaped like:

```
Got edge metrics at <microseconds> microseconds: <protobuf JSON>
```

The protobuf JSON uses the `cardio.pulseRate` field. Samples contain `value`, `confidence`, `stable`, and `timestamp` fields when available. The scorer supports both lowerCamelCase and snake_case field names and retains confidence/stability/timestamp metadata.

## Interpretation

A negative paired MAE difference favours the tested preprocessing arm. Claim an improvement only when the entire 95% confidence interval is below zero, coverage is not materially worse, and the result replicates on held-out subjects. A non-significant result is inconclusive, not proof of equivalence.

This is a research benchmark, not a medical device validation. SmartSpectra's own documentation states its metrics are for wellness/informational purposes and not for diagnosis or treatment.
