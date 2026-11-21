"""Multi-region facial rPPG extraction and signal-quality weighting.

Input frames are RGB face crops with a consistent crop/scale. This module does
not synthesize missing wavelengths or claim that an ordinary webcam captures IR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from scipy import signal


REGIONS = {
    "forehead": (0.22, 0.12, 0.78, 0.34),
    "left_cheek": (0.12, 0.43, 0.43, 0.72),
    "right_cheek": (0.57, 0.43, 0.88, 0.72),
}


@dataclass
class RegionReading:
    name: str
    rgb: np.ndarray
    quality: float
    usable: bool
    reason: str


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        raise ValueError("Monochrome/IR input needs the intensity-mode path; POS requires RGB.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an RGB frame with shape (height, width, 3).")
    if frame.dtype == np.uint8:
        return frame.astype(np.float32) / 255.0
    out = frame.astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Frame contains non-finite values.")
    if out.max(initial=0) > 1.5:
        out /= 255.0
    return np.clip(out, 0.0, 1.0)


def _region_pixels(rgb: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bounds
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    return rgb[ya:yb, xa:xb]


def _quality(pixels: np.ndarray, motion: float) -> Tuple[float, str]:
    if pixels.size == 0:
        return 0.0, "empty_region"
    brightness = float(pixels.mean())
    clipped = float(np.mean((pixels <= 0.015) | (pixels >= 0.985)))
    texture = float(np.mean(np.std(pixels, axis=(0, 1))))
    if brightness < 0.06:
        return 0.0, "underexposed"
    if brightness > 0.94 or clipped > 0.20:
        return 0.0, "overexposed_or_clipped"
    exposure_score = np.clip(1.0 - abs(brightness - 0.48) / 0.55, 0.0, 1.0)
    clipping_score = np.clip(1.0 - clipped / 0.25, 0.0, 1.0)
    texture_score = np.clip(texture / 0.035, 0.25, 1.0)
    motion_score = np.exp(-max(0.0, motion) / 3.0)
    score = float(exposure_score * clipping_score * texture_score * motion_score)
    reason = "usable" if score >= 0.08 else "low_quality"
    return score, reason


def extract_region_readings(
    frames: Iterable[np.ndarray],
    motion_limit: float = 3.0,
) -> Dict[str, RegionReading]:
    """Extract per-region RGB traces and a frame-averaged quality score."""
    seq = [_as_rgb(f) for f in frames]
    if len(seq) < 2:
        raise ValueError("At least two face frames are required.")
    shape = seq[0].shape
    if any(f.shape != shape for f in seq):
        raise ValueError("All face crops must have identical dimensions.")

    output: Dict[str, RegionReading] = {}
    for name, bounds in REGIONS.items():
        traces, scores, reasons = [], [], []
        previous_gray = None
        for frame in seq:
            pixels = _region_pixels(frame, bounds)
            traces.append(pixels.mean(axis=(0, 1)))
            gray = cv2.cvtColor(np.clip(pixels * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            motion = 0.0
            if previous_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0
                )
                motion = float(np.median(np.linalg.norm(flow, axis=2)))
            previous_gray = gray
            score, reason = _quality(pixels, motion)
            scores.append(score)
            reasons.append(reason)

        quality = float(np.mean(scores))
        usable = quality >= 0.08 and quality > 0
        reason = "usable" if usable else max(set(reasons), key=reasons.count)
        output[name] = RegionReading(
            name=name,
            rgb=np.asarray(traces, dtype=np.float32),
            quality=quality if usable else 0.0,
            usable=usable,
            reason=reason,
        )
    return output


def pos_signal(rgb_trace: np.ndarray, window: int) -> np.ndarray:
    """POS chrominance signal from a single region's RGB mean trace."""
    c = np.asarray(rgb_trace, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("rgb_trace must have shape (frames, 3).")
    if window < 2 or len(c) < window:
        raise ValueError("Not enough frames for the requested POS window.")
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    out = np.zeros(len(c), dtype=np.float64)
    counts = np.zeros(len(c), dtype=np.float64)
    for start in range(len(c) - window + 1):
        block = c[start:start + window]
        normalized = (block / (block.mean(axis=0) + 1e-8)).T
        projected = projection @ normalized
        alpha = projected[0].std() / (projected[1].std() + 1e-8)
        pulse = projected[0] + alpha * projected[1]
        pulse -= pulse.mean()
        out[start:start + window] += pulse
        counts[start:start + window] += 1.0
    return out / np.maximum(counts, 1.0)


def fuse_region_signals(
    readings: Dict[str, RegionReading],
    window: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Quality-weight POS signals; unavailable regions receive zero weight."""
    signals, weights = {}, {}
    for name, reading in readings.items():
        if reading.usable and reading.quality > 0:
            signals[name] = pos_signal(reading.rgb, window)
            weights[name] = float(reading.quality)
        else:
            weights[name] = 0.0
    total = sum(weights.values())
    if total <= 0:
        raise RuntimeError("No facial region passed the signal-quality gate.")
    normalized = {name: value / total for name, value in weights.items()}
    fused = sum(signals[name] * normalized[name] for name in signals)
    return signal.detrend(fused), normalized


def extract_intensity_trace(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Intensity trace for a real monochrome/IR camera stream."""
    values = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            raise ValueError("Intensity mode expects single-channel frames.")
        values.append(float(np.mean(arr)))
    if len(values) < 2:
        raise ValueError("At least two intensity frames are required.")
    return signal.detrend(np.asarray(values, dtype=np.float64))


def bandpass_pulse(trace: np.ndarray, fps: float, low_hz: float = 0.7, high_hz: float = 3.0) -> np.ndarray:
    """Filter a trace into a broad pulse-frequency band; not a medical reading."""
    x = np.asarray(trace, dtype=np.float64)
    if fps <= 2 * high_hz or len(x) < 16:
        raise ValueError("FPS or signal length is insufficient for this filter.")
    sos = signal.butter(3, [low_hz, high_hz], btype="bandpass", fs=fps, output="sos")
    return signal.sosfiltfilt(sos, x)


class FaceCropper:
    """MediaPipe face detector with Haar fallback and temporally stable crops.

    Detection runs on every frame. A short five-frame hold bridges detector
    dropouts without silently shifting or deleting frames from the reference.
    """
    def __init__(self, min_size: int = 48, detect_every: int = 1):
        self.min_size = int(min_size)
        self.detect_every = int(detect_every)
        self.box: Optional[Tuple[int, int, int, int]] = None
        self.frame_index = 0
        self.missed = 0
        self.mp_detector = None
        try:
            import mediapipe as mp
            self.mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.35,
            )
        except (ImportError, AttributeError):
            self.mp_detector = None

        cascade_dir = cv2.data.haarcascades
        self.cascades = [
            cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_alt2.xml"),
            cv2.CascadeClassifier(cascade_dir + "haarcascade_profileface.xml"),
        ]
        self.cascades = [c for c in self.cascades if not c.empty()]

    def _detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        if self.mp_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.mp_detector.process(rgb)
            if result.detections:
                detection = max(
                    result.detections,
                    key=lambda item: item.location_data.relative_bounding_box.width
                    * item.location_data.relative_bounding_box.height,
                )
                box = detection.location_data.relative_bounding_box
                x = max(0, int(box.xmin * width))
                y = max(0, int(box.ymin * height))
                w = min(width - x, int(box.width * width))
                h = min(height - y, int(box.height * height))
                if w >= self.min_size and h >= self.min_size:
                    return x, y, w, h

        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        candidates = []
        for detector in self.cascades:
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=4,
                minSize=(self.min_size, self.min_size),
            )
            candidates.extend(tuple(map(int, face)) for face in faces)
            if detector is self.cascades[-1]:
                flipped = cv2.flip(gray, 1)
                profiles = detector.detectMultiScale(
                    flipped, scaleFactor=1.08, minNeighbors=4,
                    minSize=(self.min_size, self.min_size),
                )
                candidates.extend(
                    (width - x - w, y, w, h) for x, y, w, h in profiles
                )
        return max(candidates, key=lambda b: b[2] * b[3]) if candidates else None

    def crop(self, bgr_frame: np.ndarray, output_size: int = 96) -> Optional[np.ndarray]:
        frame = np.asarray(bgr_frame)
        detected = self._detect(frame) if self.frame_index % self.detect_every == 0 else None
        if detected is not None:
            if self.box is None:
                self.box = detected
            else:
                self.box = tuple(
                    int(0.65 * old + 0.35 * new)
                    for old, new in zip(self.box, detected)
                )
            self.missed = 0
        else:
            self.missed += 1
            if self.missed > 5:
                self.box = None
        self.frame_index += 1
        if self.box is None:
            return None

        x, y, w, h = self.box
        side = int(max(w, h) * 1.35)
        cx, cy = x + w // 2, y + h // 2
        xa, ya = max(0, cx - side // 2), max(0, cy - side // 2)
        xb, yb = min(frame.shape[1], xa + side), min(frame.shape[0], ya + side)
        xa, ya = max(0, xb - side), max(0, yb - side)
        crop = frame[ya:yb, xa:xb]
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
