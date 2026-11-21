#!/usr/bin/env python3
"""Flicker experiment tools (pixel-domain only; the input dataset is never modified, every output is a new file).
  inject  IN OUT --strength S [--seed N]   add TV/monitor-style colour flicker (documented model below)
  nearref IN OUT                           FlyPPG near-face reference normalisation (fly-lamina-style divisive gain control)
Flicker model: gain(x,y,t) = 1 + S * f(t) * (0.35 + 0.65 * G(x,y)) * colour, f = normalised sum of 4 sinusoids in 0.6-3.8 Hz (|f|<=1),
G = centred Gaussian (the person is lit more than the corners, like a screen in front of them), colour = R .55 / G .8 / B 1.4.
Near-face reference: per frame, per channel, mean of the non-skin ring just outside the face (hair/ears/sides, above the chin) tracked with
MediaPipe face-mesh landmarks. The coupling beta (0..1.2) between the cheek-ROI illumination and that ring is regressed per channel in the
0.1-5 Hz band, and only the coupled part is divided out, so a ring that does not track the face's light (e.g. moving hair) changes nothing."""
import argparse, subprocess, sys
import cv2, numpy as np
from scipy import ndimage as ndi
import imageio_ffmpeg

LC = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]; RC = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]
OVAL_EDGES = None


def reader(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; W, H = int(cap.get(3)), int(cap.get(4)); return cap, fps, W, H


def writer(path, fps, W, H):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                             "-an", "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


def flicker_signal(n, fps, seed):
    r = np.random.default_rng(seed); t = np.arange(n) / fps
    f = sum(np.sin(2 * np.pi * r.uniform(.6, 3.8) * t + r.uniform(0, 6.28)) * r.uniform(.4, 1) for _ in range(4)); return f / np.abs(f).max()


def inject(src, dst, strength, seed=1):
    cap, fps, W, H = reader(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); f = flicker_signal(n + 10, fps, seed)
    yy, xx = np.mgrid[:H, :W].astype(np.float32); G = np.exp(-(((xx - W / 2) / (.30 * W)) ** 2 + ((yy - H / 2) / (.35 * H)) ** 2) / 2)
    field = (.35 + .65 * G)[..., None]; color = np.array([1.4, .8, .55], np.float32)   # BGR order: B 1.4, G .8, R .55
    out = writer(dst, fps, W, H); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        gain = 1 + strength * f[i] * field * color
        out.stdin.write(np.clip(fr.astype(np.float32) * gain, 0, 255).astype(np.uint8).tobytes()); i += 1
    cap.release(); out.stdin.close(); out.wait()
    print(f"injected strength={strength} into {i} frames, flicker band 0.6-3.8 Hz")


def _mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=.3, min_tracking_confidence=.3), \
        sorted({i for e in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in e})


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
    bp = lambda x: ndi.gaussian_filter1d(x, 3, axis=0) - ndi.gaussian_filter1d(x, 30, axis=0); bg_, bf = bp(g), bp(f)
    beta = np.clip((bg_ * bf).sum(0) / ((bg_ ** 2).sum(0) + 1e-9), 0, 1.2)            # per channel (B, G, R)
    gain = np.exp(-beta * (g - np.median(g, 0)))                                     # divide out only the coupled part
    print(f"near-face reference: frames={frames_small} landmark_found={sum(x is not None for x in ring)} beta(BGR)={np.round(beta, 3).tolist()}")
    cap, fps, W, H = reader(src); out = writer(dst, fps, W, H); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        out.stdin.write(np.clip(fr.astype(np.float32) * gain[min(i, len(gain) - 1)].astype(np.float32), 0, 255).astype(np.uint8).tobytes()); i += 1
    cap.release(); out.stdin.close(); out.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["inject", "nearref"]); ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--strength", type=float, default=0.1); ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
    inject(a.src, a.dst, a.strength, a.seed) if a.cmd == "inject" else nearref(a.src, a.dst)
