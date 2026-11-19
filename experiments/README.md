# First real-face benchmark: UBFC-rPPG

This is the first real-video experiment for FlyPPG. It deliberately starts with one recording to validate the complete video/reference alignment path before spending bandwidth on the full dataset.

## Data

Use one authorized UBFC-rPPG subject folder containing:
- `vid.avi`
- `ground_truth.txt`

The ground-truth file has three rows: reference PPG, instantaneous HR, and timestamps. Respect the dataset's research terms and participant restrictions. Do not publish raw face footage or extracted face crops.

Official dataset page: https://sites.google.com/view/ybenezeth/ubfcrppg

## Run

```bash
python -m pip install numpy scipy opencv-python
python experiments/ubfc_single.py --video /path/to/subject1/vid.avi --ground-truth /path/to/subject1/ground_truth.txt --out results/subject1
```

The script reports raw-video POS, surround-normalised POS, and FlyPPG-v3 stabilised POS when `readout_W3.npy` is present. It emits per-window estimates in `windows.csv` and a summary in `report.json`.

## Interpretation limits

- This is a pipeline smoke test, not a publishable benchmark.
- Haar face tracking and a fixed central face crop are a simple, reproducible ROI baseline; detection failures and crop drift must be audited.
- POS is applied identically to each preprocessing arm. This isolates the effect of preprocessing rather than changing the pulse extractor.
- The current script uses 10-second windows with 5-second stride. The reference HR is averaged over each matching time interval.
- SmartSpectra is not run by this script. It requires the licensed SDK and key. Use the existing `ss_preprocess.py`, `ss_run.py`, and `ss_score.py` path for the SDK arm, then add its timestamped outputs to the same window-level scoring protocol.
- One subject cannot establish generalisation. Repeat across held-out participants and report coverage, MAE, RMSE and paired per-subject uncertainty.
- No results are claimed until a real video and its reference file have actually been processed.
