"""Compare raw-face POS, each facial region, and quality-weighted fusion."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facial_signal import (
    FaceCropper, REGIONS, _region_pixels, bandpass_pulse,
    extract_region_readings, fuse_region_signals, pos_signal,
)


def dominant_bpm(values, fps):
    x = bandpass_pulse(values, fps)
    f, p = signal.welch(x, fs=fps, nperseg=len(x), nfft=max(4096, len(x) * 4))
    band = (f >= 0.7) & (f <= 3.0)
    return float(f[band][np.argmax(p[band])] * 60.0)


def spectral_peak(values, fps):
    """Return dominant BPM and its fraction of in-band spectral power."""
    x = bandpass_pulse(values, fps)
    frequencies, power = signal.welch(
        x, fs=fps, nperseg=len(x), nfft=max(4096, len(x) * 4)
    )
    band = (frequencies >= 0.7) & (frequencies <= 3.0)
    band_power = power[band]
    if not len(band_power) or float(band_power.sum()) <= 0:
        return None, 0.0
    peak_index = int(np.argmax(band_power))
    bpm = float(frequencies[band][peak_index] * 60.0)
    concentration = float(band_power[peak_index] / band_power.sum())
    return bpm, concentration


def weighted_median(values, weights):
    """Weighted median that does not average incompatible regional BPMs."""
    pairs = sorted(
        (float(value), float(weight))
        for value, weight in zip(values, weights)
        if value is not None and np.isfinite(value) and weight > 0
    )
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    halfway = total / 2.0
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= halfway:
            return value
    return pairs[-1][0]


def load_face_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 10:
        cap.release()
        raise RuntimeError(f"Invalid or unsuitable video FPS: {fps}")
    cropper = FaceCropper()
    crops = []
    missing = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = cropper.crop(frame)
        if crop is None:
            missing += 1
            crops.append(None)
        else:
            crops.append(crop)
    cap.release()
    elapsed = time.perf_counter() - started
    if missing:
        raise RuntimeError(
            f"Face detection failed on {missing}/{len(crops)} frames. "
            "No dropping frames: reference alignment must remain exact."
        )
    return crops, fps, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--stride", type=float, default=5.0)
    args = parser.parse_args()

    total_start = time.perf_counter()
    frames, fps, crop_seconds = load_face_video(args.video)
    reference = np.loadtxt(args.reference)[:, 0]
    if len(frames) != len(reference):
        raise RuntimeError(f"Frame/reference mismatch: {len(frames)} vs {len(reference)}")

    started = time.perf_counter()
    readings = extract_region_readings(frames)
    region_seconds = time.perf_counter() - started

    pos_window = max(2, round(1.6 * fps))
    raw_rgb = np.asarray([frame.mean(axis=(0, 1)) for frame in frames])
    methods = {"raw_face_POS": pos_signal(raw_rgb, pos_window)}
    for name, bounds in REGIONS.items():
        region_rgb = np.asarray([
            _region_pixels(frame, bounds).mean(axis=(0, 1)) for frame in frames
        ])
        methods[name + "_POS"] = pos_signal(region_rgb, pos_window)

    fused, weights = fuse_region_signals(readings, window=pos_window)
    methods["quality_weighted_fusion"] = fused

    size = round(args.window * fps)
    step = round(args.stride * fps)
    rows = []
    for start in range(0, len(frames) - size + 1, step):
        stop = start + size
        truth = dominant_bpm(reference[start:stop], fps)
        row = {"center_s": (start + stop - 1) / (2 * fps), "reference_bpm": truth}
        for name, trace in methods.items():
            estimate = dominant_bpm(trace[start:stop], fps)
            row[name + "_bpm"] = estimate
            row[name + "_absolute_error_bpm"] = abs(estimate - truth)

        # Estimate each facial region independently, then combine BPMs rather
        # than waveforms. This avoids phase cancellation between regions.
        regional_bpms, regional_weights = [], []
        regional_diagnostics = {}
        for region in ("forehead", "left_cheek", "right_cheek"):
            estimate, concentration = spectral_peak(
                methods[region + "_POS"][start:stop], fps
            )
            quality = readings[region].quality if readings[region].usable else 0.0
            weight = quality * concentration
            regional_bpms.append(estimate)
            regional_weights.append(weight)
            regional_diagnostics[region] = {
                "bpm": estimate,
                "spectral_concentration": concentration,
                "weight": weight,
            }
        consensus = weighted_median(regional_bpms, regional_weights)
        row["region_consensus_bpm"] = consensus
        row["region_consensus_absolute_error_bpm"] = (
            abs(consensus - truth) if consensus is not None else None
        )
        row["region_consensus_regions"] = regional_diagnostics
        rows.append(row)

    metrics = {}
    for name in methods:
        errors = np.asarray([r[name + "_absolute_error_bpm"] for r in rows])
        metrics[name] = {
            "mae_bpm": float(errors.mean()) if len(errors) else None,
            "rmse_bpm": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            "within_5_bpm_pct": float(np.mean(errors <= 5) * 100) if len(errors) else None,
        }

    consensus_errors = np.asarray([
        row["region_consensus_absolute_error_bpm"]
        for row in rows
        if row["region_consensus_absolute_error_bpm"] is not None
    ], dtype=np.float64)
    metrics["region_consensus"] = {
        "mae_bpm": float(consensus_errors.mean()) if len(consensus_errors) else None,
        "rmse_bpm": float(np.sqrt(np.mean(consensus_errors ** 2))) if len(consensus_errors) else None,
        "within_5_bpm_pct": float(np.mean(consensus_errors <= 5) * 100) if len(consensus_errors) else None,
    }

    report = {
        "recording": Path(args.video).name,
        "fps": fps,
        "frames": len(frames),
        "windows_scored": len(rows),
        "metrics": metrics,
        "region_quality": {
            name: {
                "quality": reading.quality,
                "usable": reading.usable,
                "reason": reading.reason,
            } for name, reading in readings.items()
        },
        "fusion_weights": weights,
        "runtime_seconds": {
            "face_detection_and_crop": crop_seconds,
            "region_quality_and_flow": region_seconds,
            "total": time.perf_counter() - total_start,
        },
        "windows": rows,
        "limitations": [
            "Overlapping windows are not statistically independent.",
            "Face detection must succeed on every frame to preserve reference alignment.",
            "Research benchmark only; not a medical measurement.",
            "Monochrome/IR intensity path is not evaluated by this RGB benchmark.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "metrics": metrics,
        "region_quality": report["region_quality"],
        "fusion_weights": weights,
        "runtime_seconds": report["runtime_seconds"],
        "windows_scored": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
