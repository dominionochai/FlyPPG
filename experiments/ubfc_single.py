"""Run a one-video, real-data UBFC-rPPG benchmark.

Extracts a tracked face crop, applies POS to raw and FlyPPG-preprocessed frames,
and compares windowed pulse estimates with the synchronized reference HR series.
This does not run SmartSpectra; use ss_preprocess.py + ss_run.py for that SDK arm.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

FPS_FALLBACK = 30.0
FACE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def parse_ground_truth(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("Expected UBFC ground_truth.txt with PPG, HR and timestamps rows")
    hr = np.fromstring(lines[1], sep=" ")
    stamps = np.fromstring(lines[2], sep=" ")
    if len(hr) < 2 or len(stamps) != len(hr):
        raise ValueError("Reference HR and timestamp rows are missing or mismatched")
    stamps = stamps - stamps[0]
    if np.nanmax(stamps) > 1e5:
        stamps = stamps / 1e6
    elif np.nanmax(stamps) > 1e2:
        stamps = stamps / 1e3
    return stamps.astype(float), hr.astype(float)


def extract_face_video(path, limit_seconds=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    frames, last_box = [], None
    frame_index = 0
    max_frames = int(limit_seconds * fps) if limit_seconds else None
    height = width = 0
    while max_frames is None or frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(80, 80))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            last_box = (x, y, w, h)
        if last_box is None:
            # Keep the original frame index; use a conservative central ROI
            # until the detector acquires a face rather than shifting timestamps.
            side = min(height, width) // 2
            x = (width - side) // 2
            y = (height - side) // 2
            w = h = side
        else:
            x, y, w, h = last_box
        # Preserve facial context and some surrounding background: FlyPPG's
        # surround normaliser requires both, unlike a face-only cheek crop.
        side = int(max(w, h) * 1.8)
        cx, cy = x + w // 2, y + h // 2
        xa, ya = max(0, cx - side // 2), max(0, cy - side // 2)
        xb, yb = min(width, xa + side), min(height, ya + side)
        xa, ya = max(0, xb - side), max(0, yb - side)
        crop = frame[ya:yb, xa:xb]
        if crop.size == 0:
            raise RuntimeError(f"Invalid crop at video frame {frame_index}")
        rgb = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2RGB)
        frames.append(rgb.astype(np.float32) / 255.0)
    cap.release()
    if len(frames) < int(fps * 12):
        raise RuntimeError("Could not extract at least 12 seconds of usable face video")
    return np.asarray(frames, dtype=np.float32), float(fps)


def pos_signal(frames, fps, window_seconds=1.6):
    n = len(frames)
    win = max(2, int(round(window_seconds * fps)))
    rgb = frames.mean(axis=(1, 2))
    pulse = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    for start in range(0, n - win + 1):
        chunk = rgb[start:start + win]
        mean = chunk.mean(axis=0)
        if np.any(mean <= 1e-8):
            continue
        normalized = (chunk / mean) * 100.0
        s1 = normalized[:, 1] - normalized[:, 2]
        s2 = normalized[:, 1] + normalized[:, 2] - 2 * normalized[:, 0]
        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        h = s1 + alpha * s2
        h -= h.mean()
        pulse[start:start + win] += h
        weights[start:start + win] += 1
    return pulse / np.maximum(weights, 1)


def estimate_windows(pulse, fps, ref_t, ref_hr, window=10.0, stride=5.0):
    low, high = 0.7, 3.0
    b, a = signal.butter(3, [low, high], btype="bandpass", fs=fps)
    rows = []
    size = int(round(window * fps))
    step = max(1, int(round(stride * fps)))
    for start in range(0, len(pulse) - size + 1, step):
        stop = start + size
        x = signal.detrend(pulse[start:stop])
        try:
            x = signal.filtfilt(b, a, x)
        except ValueError:
            continue
        freq, power = signal.welch(x, fs=fps, nperseg=min(size, len(x)),
                                   nfft=max(4096, size))
        band = (freq >= low) & (freq <= high)
        if not np.any(band):
            continue
        est = float(freq[band][np.argmax(power[band])] * 60.0)
        center = (start + stop - 1) / (2 * fps)
        valid = (ref_t >= center - window / 2) & (ref_t <= center + window / 2)
        if not np.any(valid):
            truth = float(np.interp(center, ref_t, ref_hr))
        else:
            truth = float(np.mean(ref_hr[valid]))
        rows.append({"start_s": start / fps, "center_s": center,
                     "reference_bpm": truth, "estimate_bpm": est,
                     "absolute_error_bpm": abs(est - truth)})
    return rows


def summarize(rows):
    if not rows:
        return {"windows": 0, "mae_bpm": None, "rmse_bpm": None,
                "within_5_bpm_pct": None}
    errors = np.asarray([r["estimate_bpm"] - r["reference_bpm"] for r in rows])
    return {
        "windows": int(len(errors)),
        "mae_bpm": float(np.mean(np.abs(errors))),
        "rmse_bpm": float(np.sqrt(np.mean(errors ** 2))),
        "within_5_bpm_pct": float(np.mean(np.abs(errors) <= 5) * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", default="ubfc_single_results")
    ap.add_argument("--limit-seconds", type=float)
    ap.add_argument("--flyppg-weights", default="readout_W3.npy")
    args = ap.parse_args()

    frames, fps = extract_face_video(args.video, args.limit_seconds)
    ref_t, ref_hr = parse_ground_truth(args.ground_truth)
    import flyppg3 as fly

    # Use the repository's own POS implementation consistently across arms.
    arms = {"raw_POS": frames}
    arms["surround_POS"] = fly.surround_gated(frames)
    weights_path = Path(args.flyppg_weights)
    if weights_path.exists():
        weights = np.load(weights_path)
        arms["flyppg_v3_POS"] = fly.stab_fly_local(frames, weights)
    else:
        print("WARNING: readout_W3.npy missing; FlyPPG v3 arm skipped")

    os.makedirs(args.out, exist_ok=True)
    report = {
        "dataset": "UBFC-rPPG",
        "video": Path(args.video).name,
        "fps": fps,
        "frames": int(len(frames)),
        "duration_seconds": len(frames) / fps,
        "face_crop": "OpenCV Haar face box, central cheek/forehead crop, 96x96",
        "reference": "synchronized finger-oximeter HR series",
        "smart_spectra": "NOT RUN by this script",
        "arms": {},
    }
    with open(Path(args.out) / "windows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "arm", "start_s", "center_s", "reference_bpm",
            "estimate_bpm", "absolute_error_bpm"])
        writer.writeheader()
        for name, video in arms.items():
            pulse = fly.rppg_pos(video, l=max(2, int(round(1.6 * fps))))
            rows = estimate_windows(pulse, fps, ref_t, ref_hr)
            report["arms"][name] = summarize(rows)
            for row in rows:
                writer.writerow({"arm": name, **row})
    Path(args.out, "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
