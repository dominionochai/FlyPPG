"""Landmark-tracked skin regions with per-frame visibility, fused by visibility x spectral SNR.

A Haar crop loses the face on a head turn; MediaPipe FaceMesh keeps tracking, so the
region that turns away is down-weighted instead of the whole pulse estimate collapsing.
"""
import cv2
import numpy as np
from scipy import signal

from facial_signal import bandpass_pulse, pos_signal

REGIONS = {  # FaceMesh indices; polygons checked on skin (forehead stays below the hairline)
    "forehead": [104, 69, 108, 151, 337, 299, 333, 334, 9, 107, 105],
    "left_cheek": [118, 119, 100, 126, 209, 49, 129, 203, 205, 50, 101],
    "right_cheek": [347, 348, 329, 355, 429, 279, 358, 423, 425, 280, 330],
}
NAMES = list(REGIONS)
EDGE_L, EDGE_R, NOSE = 234, 454, 1


def cheek_visibility(x_left, x_right, x_nose):
    """(left, right) cheek visibility from where the nose sits between the face edges.

    Frontal: nose is centred -> (1, 1). Turning moves the nose toward the receding
    edge, shrinking that cheek's share and driving it to 0 at profile.
    """
    dl, dr = x_nose - x_left, x_right - x_nose
    total = max(dl + dr, 1e-6)
    return tuple(float(np.clip(2 * d / total, 0, 1) ** 2) for d in (dl, dr))


class LandmarkTracker:
    """Per frame: mean RGB of each region (NaN if no face) and its visibility."""

    def __init__(self):
        from mediapipe.python.solutions import face_mesh  # pinned: mediapipe==0.10.14
        self.mesh = face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        found = self.mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).multi_face_landmarks
        if not found:
            return np.full((len(NAMES), 3), np.nan), np.zeros(len(NAMES))
        p = np.array([[l.x * w, l.y * h] for l in found[0].landmark])
        rgb = np.empty((len(NAMES), 3))
        for i, idx in enumerate(REGIONS.values()):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(p[idx].astype(np.int32)), 255)
            rgb[i] = cv2.mean(bgr, mask)[2::-1]
        vl, vr = cheek_visibility(p[EDGE_L, 0], p[EDGE_R, 0], p[NOSE, 0])
        return rgb, np.array([1.0, vl, vr])


def track_video(path):
    """-> rgb (T, R, 3) with NaN gaps, vis (T, R), fps. Never drops a frame."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker, rgb, vis = LandmarkTracker(), [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r, v = tracker(frame)
        rgb.append(r), vis.append(v)
    cap.release()
    return np.array(rgb), np.array(vis), float(fps)


def fill_gaps(x):
    """Linearly interpolate NaN rows of (T, ...) so POS sees a continuous trace."""
    x = x.copy()
    bad = np.isnan(x).reshape(len(x), -1).any(1)
    if bad.all():
        raise RuntimeError("No face found in any frame.")
    t = np.arange(len(x))
    for idx in np.ndindex(*x.shape[1:]):
        col = (slice(None),) + idx
        x[col][bad] = np.interp(t[bad], t[~bad], x[col][~bad])
    return x


def dominant_bpm(x, fps):
    f, p = signal.welch(bandpass_pulse(x, fps), fs=fps, nperseg=len(x), nfft=max(4096, 4 * len(x)))
    band = (f >= 0.7) & (f <= 3.0)
    return float(f[band][p[band].argmax()] * 60)


def snr(x, fps):
    """Share of in-band power within 0.12 Hz of the spectral peak (0..1)."""
    f, p = signal.welch(bandpass_pulse(x, fps), fs=fps, nperseg=len(x), nfft=max(4096, 4 * len(x)))
    band = (f >= 0.7) & (f <= 3.0)
    f, p = f[band], p[band]
    return float(p[abs(f - f[p.argmax()]) <= 0.12].sum() / p.sum())


def fuse(sigs, vis, fps):
    """Fuse (R, n) POS segments. Weight = mean visibility x SNR; hidden frames add nothing."""
    z = (sigs - sigs.mean(1, keepdims=True)) / (sigs.std(1, keepdims=True) + 1e-9)
    w = vis.mean(1) * np.array([snr(s, fps) for s in sigs])
    per_frame = vis * w[:, None]
    return (per_frame * z).sum(0) / np.maximum(per_frame.sum(0), 1e-9), w / max(w.sum(), 1e-9)


def region_pos(rgb, fps):
    """POS signal per region from gap-filled (T, R, 3) traces -> (R, T)."""
    filled = fill_gaps(rgb)
    win = max(2, round(1.6 * fps))
    return np.array([pos_signal(filled[:, i], win) for i in range(filled.shape[1])])
