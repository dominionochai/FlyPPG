"""Evaluate the multi-region facial signal module against synchronized MCD PPG."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

from facial_signal import FaceCropper, extract_region_readings, fuse_region_signals, bandpass_pulse


def dominant_bpm(values, fps):
    x = bandpass_pulse(values, fps)
    f, p = signal.welch(x, fs=fps, nperseg=len(x), nfft=max(4096, len(x) * 4))
    band = (f >= 0.7) & (f <= 3.0)
    return float(f[band][np.argmax(p[band])] * 60.0)


def load_face_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 10:
        cap.release()
        raise RuntimeError(f"Invalid or unsuitable video FPS: {fps}")
    cropper = FaceCropper()
    crops, missing = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = cropper.crop(frame)
        if crop is None:
            missing += 1
        else:
            crops.append(crop)
    cap.release()
    if missing:
        raise RuntimeError(
            f"Face detection failed on {missing} frames. Refusing to score a "
            "temporally misaligned subset; inspect detector/crop quality first."
        )
    return crops, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--stride", type=float, default=5.0)
    args = parser.parse_args()

    frames, fps = load_face_video(args.video)
    reference = np.loadtxt(args.reference)[:, 0]
    if len(frames) != len(reference):
        raise RuntimeError(
            f"Frame/reference mismatch: {len(frames)} frames vs {len(reference)} samples"
        )

    readings = extract_region_readings(frames)
    fused, weights = fuse_region_signals(readings, window=max(2, round(1.6 * fps)))

    size = round(args.window * fps)
    step = round(args.stride * fps)
    rows = []
    for start in range(0, len(frames) - size + 1, step):
        stop = start + size
        truth = dominant_bpm(reference[start:stop], fps)
        estimate = dominant_bpm(fused[start:stop], fps)
        rows.append({
            "center_s": (start + stop - 1) / (2 * fps),
            "reference_bpm": truth,
            "multiregion_bpm": estimate,
            "absolute_error_bpm": abs(estimate - truth),
        })

    errors = np.asarray([row["absolute_error_bpm"] for row in rows], dtype=float)
    report = {
        "recording": Path(args.video).name,
        "fps": fps,
        "frames": len(frames),
        "regions": {
            name: {
                "quality": reading.quality,
                "usable": reading.usable,
                "reason": reading.reason,
            }
            for name, reading in readings.items()
        },
        "fusion_weights": weights,
        "windows_scored": len(rows),
        "mae_bpm": float(errors.mean()) if len(errors) else None,
        "within_5_bpm_pct": float(np.mean(errors <= 5) * 100) if len(errors) else None,
        "windows": rows,
        "limitations": [
            "Research benchmark only; not a medical measurement.",
            "Face detection must succeed on every frame to preserve reference alignment.",
            "IR intensity mode is not evaluated by this RGB POS benchmark.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in (
        "regions", "fusion_weights", "windows_scored", "mae_bpm", "within_5_bpm_pct"
    )}, indent=2))


if __name__ == "__main__":
    main()
