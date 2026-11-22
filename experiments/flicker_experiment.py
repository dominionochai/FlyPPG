#!/usr/bin/env python3
"""Flicker experiment tools (pixel-domain only; the input dataset is never modified, every output is a new file).
  inject  IN OUT --strength S [--seed N] [--freq-bpm B]   add TV/monitor-style colour flicker (documented model below); with --freq-bpm
                                                          the flicker is a single sinusoid at exactly B beats/min (entrainment trap: a
                                                          pulse-rate estimate that reports B is locked to the flicker, not to the heart)
  nearref IN OUT                           FlyPPG near-face reference normalisation (fly-lamina-style divisive gain control)
  denotch IN OUT                           gated global flicker cancellation: per-channel log-mean brightness is bandpassed (0.5-4 Hz)
                                           and the whole common-mode series is divided out. Engages only when the band energy exceeds
                                           pulse level (~2x worst clean pulse amplitude), so on clean video it is a no-op by design
Flicker model: gain(x,y,t) = 1 + S * f(t) * (0.35 + 0.65 * G(x,y)) * colour, f = normalised sum of 4 sinusoids in 0.6-3.8 Hz (|f|<=1),
G = centred Gaussian (the person is lit more than the corners, like a screen in front of them), colour = R .55 / G .8 / B 1.4.
Near-face reference: per frame, per channel, mean of the non-skin ring just outside the face (hair/ears/sides, above the chin) tracked with
MediaPipe face-mesh landmarks. The coupling beta (0..1.2) between the cheek-ROI illumination and that ring is regressed per channel in the
0.1-5 Hz band, and only the coupled part is divided out, so a ring that does not track the face's light (e.g. moving hair) changes nothing."""
import argparse, subprocess, sys
import cv2, numpy as np
import imageio_ffmpeg

PULSEAMP = 0.004
MAX_CORR = 0.30
RING_CORR = 0.75

LC = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]; RC = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]
OVAL_EDGES = None


def reader(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; W, H = int(cap.get(3)), int(cap.get(4)); return cap, fps, W, H


def writer(path, fps, W, H):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                             "-an", "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


def flicker_signal(n, fps, seed, freq_bpm=None):
    r = np.random.default_rng(seed); t = np.arange(n) / fps
    if freq_bpm:
        return np.sin(2 * np.pi * (freq_bpm / 60.0) * t)
    f = sum(np.sin(2 * np.pi * r.uniform(.6, 3.8) * t + r.uniform(0, 6.28)) * r.uniform(.4, 1) for _ in range(4)); return f / np.abs(f).max()


def inject(src, dst, strength, seed=1, freq_bpm=None):
    cap, fps, W, H = reader(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); f = flicker_signal(n + 10, fps, seed, freq_bpm)
    yy, xx = np.mgrid[:H, :W].astype(np.float32); G = np.exp(-(((xx - W / 2) / (.30 * W)) ** 2 + ((yy - H / 2) / (.35 * H)) ** 2) / 2)
    field = (.35 + .65 * G)[..., None]; color = np.array([1.4, .8, .55], np.float32)   # BGR order: B 1.4, G .8, R .55
    out = writer(dst, fps, W, H); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        gain = 1 + strength * f[i] * field * color
        out.stdin.write(np.clip(fr.astype(np.float32) * gain, 0, 255).astype(np.uint8).tobytes()); i += 1
    cap.release(); out.stdin.close(); out.wait()
    label = f"flicker at {freq_bpm} bpm" if freq_bpm else "flicker band 0.6-3.8 Hz"; print(f"injected strength={strength} into {i} frames, {label}")


def _mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=.3, min_tracking_confidence=.3), \
        sorted({i for e in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in e})


def bandpass_series(x, fps):
    """Bandpass a 1-D series to the ~0.5-4 Hz flicker band.

    FFT mask with smooth tanh edges: exact, zero-phase and idempotent (band(band(x)) == band(x)),
    unlike a Gaussian difference which phase-inverts above its geometric-mean cutoff and would
    re-add the very flicker it is meant to cancel.
    """
    x = np.asarray(x, np.float64); n = len(x)
    if n < 8 or fps <= 0:
        return np.zeros_like(x)
    fr = np.fft.rfftfreq(n, 1.0 / fps)
    H = 0.5 * (1 + np.tanh((fr - 0.5) / 0.15)) * 0.5 * (1 - np.tanh((fr - 4.0) / 0.6))
    return np.fft.irfft(np.fft.rfft(x - x.mean()) * H, n)


def dominant_freq(x, fps):
    x = np.asarray(x, np.float64); n = len(x)
    if n < 8 or fps <= 0:
        return 0.0, 0.0
    f = np.fft.rfftfreq(n, 1.0 / fps); p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    band = (f >= 0.5) & (f <= 4.0)
    if not band.any() or p[band].sum() <= 0:
        return 0.0, 0.0
    i = np.flatnonzero(band)[np.argmax(p[band])]
    return float(f[i]), float(p[i] / max(p[band].sum(), 1e-12))


def denotch_gains(means, fps):
    """means: (T,3) per-frame per-channel linear means of the WHOLE frame.

    Returns (gain (T,3), info). In log domain the bandpass isolates common-mode flicker; the gain
    divides it out per channel. Arming gate: the median band std over channels must exceed PULSEAMP
    (clean video sits an order of magnitude below) and the dominant frequency must concentrate
    >= 15% of in-band power (both real flicker models clear this; broadband noise/pulse does not).
    A disarmed gate returns an exact identity gain so clean clips are untouched.
    """
    m = np.asarray(means, np.float64)
    info = {"armed": False, "freq_hz": 0.0, "concentration": 0.0, "band_std": [0.0, 0.0, 0.0]}
    T = len(m)
    if T < 30:
        return np.ones((T, 3), np.float32), info
    g = np.log(np.maximum(m, 1e-6))
    f = np.stack([bandpass_series(g[:, c], fps) for c in range(3)], 1)
    stds = f.std(0)
    info["band_std"] = [round(float(s), 5) for s in stds]
    freq, conc = dominant_freq(f.mean(1), fps)
    info["freq_hz"], info["concentration"] = round(freq, 3), round(conc, 3)
    if float(np.median(stds)) <= PULSEAMP or conc < 0.15 or freq <= 0:
        return np.ones((T, 3), np.float32), info
    info["armed"] = True
    corr = np.exp(np.clip(-(f - f.mean(0)), -MAX_CORR, MAX_CORR))
    return corr.astype(np.float32), info


def denotch(src, dst):
    cap, fps, W, H = reader(src); sc = min(1.0, 320.0 / max(W, H)); sw, sh = max(2, int(W * sc)), max(2, int(H * sc))
    means = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        means.append(cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA).reshape(-1, 3).mean(0))
    cap.release()
    gain, info = denotch_gains(means, fps)
    print(f"denotch: armed={info['armed']} dom_freq={info['freq_hz']}Hz concentration={info['concentration']} band_std(BGR)={info['band_std']}")
    cap, fps, W, H = reader(src); out = writer(dst, fps, W, H); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        out.stdin.write(np.clip(fr.astype(np.float32) * gain[min(i, len(gain) - 1)], 0, 255).astype(np.uint8).tobytes()); i += 1
    cap.release(); out.stdin.close(); out.wait()


def nearref(src, dst):
    cap, fps, W, H = reader(src); sc = min(1.0, 320.0 / max(W, H)); sw, sh = max(2, int(W * sc)), max(2, int(H * sc))
    fm, OVAL = _mesh(); ring, roi, last, frames_small = [], [], None, 0
    # pass 1: landmarks on a downscaled copy -> per-frame ring mean and cheek-ROI mean (log domain later)
    while True:
        ok, fr = cap.read()
        if not ok: break
        small = cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA)
        r = fm.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if r.multi_face_landmarks: last = np.array([[p.x * sw, p.y * sh] for p in r.multi_face_landmarks[0].landmark[:468]], np.float32)
        frames_small += 1
        if last is None: ring.append(None); roi.append(None); continue
        P = last[OVAL]; c = P.mean(0)
        big = cv2.convexHull(((P - c) * 1.42 + c).astype(np.float32)).reshape(-1, 2); sm_ = cv2.convexHull(((P - c) * 1.12 + c).astype(np.float32)).reshape(-1, 2)
        m = np.zeros((sh, sw), np.uint8); cv2.fillConvexPoly(m, big.astype(np.int32), 1); cv2.fillConvexPoly(m, sm_.astype(np.int32), 0); m[int(last[152, 1]) - 2:, :] = 0
        mc = np.zeros((sh, sw), np.uint8)
        for idx in (LC, RC): cv2.fillConvexPoly(mc, last[idx].astype(np.int32), 1)
        ring.append(small[m.astype(bool)].mean(0) if m.sum() > 30 else None); roi.append(small[mc.astype(bool)].mean(0) if mc.sum() > 10 else None)
    cap.release()
    def fill(a):
        out, prev = [], None
        for x in a:
            if x is not None: prev = x
            out.append(prev)
        first = next((x for x in out if x is not None), np.ones(3, np.float32)); return np.array([x if x is not None else first for x in out], np.float64)
    g, f = np.log(fill(ring) + 1e-6), np.log(fill(roi) + 1e-6)
    bp = lambda x: bandpass_series(x, fps); bg_, bf = bp(g), bp(f)
    beta = np.clip((bg_ * bf).sum(0) / ((bg_ ** 2).sum(0) + 1e-9), 0, 1.2)            # per channel (B, G, R)
    ringc, roic = np.stack([np.corrcoef(bg_[:, c], bf[:, c])[0, 1] for c in range(3)]), bf.std(0)
    coupled = np.isfinite(ringc).all() and float(np.median(ringc)) > RING_CORR and float(np.median(roic)) > PULSEAMP
    beta = beta if coupled else np.zeros(3)                                          # no verified ring<->face flicker coupling: exact no-op
    gain = np.exp(-beta * (g - np.median(g, 0)))                                     # divide out only the coupled part
    print(f"near-face reference: frames={frames_small} landmark_found={sum(x is not None for x in ring)} armed={coupled} "
          f"ring_corr(BGR)={np.round(ringc, 3).tolist()} roi_band_std(BGR)={np.round(roic, 5).tolist()} beta(BGR)={np.round(beta, 3).tolist()}")
    cap, fps, W, H = reader(src); out = writer(dst, fps, W, H); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        out.stdin.write(np.clip(fr.astype(np.float32) * gain[min(i, len(gain) - 1)].astype(np.float32), 0, 255).astype(np.uint8).tobytes()); i += 1
    cap.release(); out.stdin.close(); out.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["inject", "nearref", "denotch"]); ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--strength", type=float, default=0.1); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--freq-bpm", type=float, default=None); a = ap.parse_args()
    inject(a.src, a.dst, a.strength, a.seed, a.freq_bpm) if a.cmd == "inject" else (nearref(a.src, a.dst) if a.cmd == "nearref" else denotch(a.src, a.dst))
