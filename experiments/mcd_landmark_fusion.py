"""Landmark-tracked regions: each region alone vs visibility-weighted fusion, on the MCD clip.

--stress simulates turning: spans where one region carries pulse-free noise and reports
zero visibility (what the tracker reports when that cheek turns away). Synthetic; it tests
the fusion mechanism, not the accuracy of the visibility estimate.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landmark_regions import NAMES, dominant_bpm, fuse, region_pos, track_video


def occlude(rgb, vis, fps, span_s=10, seed=0):
    """Alternate left/right-cheek loss every span_s: replace trace with pulse-free noise, vis=0."""
    rng, rgb, vis = np.random.default_rng(seed), rgb.copy(), vis.copy()
    span = round(span_s * fps)
    for k, a in enumerate(range(0, len(rgb), span)):
        r = 1 + k % 2
        n = len(rgb[a:a + span])
        drift = np.cumsum(rng.normal(0, 0.3, (n, 3)), 0)
        rgb[a:a + span, r] = np.nanmean(rgb[:, r], 0) + drift
        vis[a:a + span, r] = 0
    return rgb, vis


def score(methods, truth_fn, fps, n, win, step):
    rows = []
    for a in range(0, n - win + 1, step):
        truth = truth_fn(a, a + win)
        rows.append({"t": a / fps, "ref": truth, **{m: abs(f(a, a + win) - truth) for m, f in methods.items()}})
    out = {}
    for m in methods:
        e = np.array([r[m] for r in rows])
        out[m] = {"mae": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                  "within5_pct": float((e <= 5).mean() * 100)}
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=10)
    ap.add_argument("--stride", type=float, default=5)
    a = ap.parse_args()

    rgb, vis, fps = track_video(a.video)
    ref = np.loadtxt(a.reference)[:, 0]
    if len(rgb) != len(ref):
        raise RuntimeError(f"Frame/reference mismatch: {len(rgb)} vs {len(ref)}")
    report = {"fps": fps, "frames": len(rgb), "frames_no_face": int(np.isnan(rgb[:, 0, 0]).sum()),
              "mean_visibility": dict(zip(NAMES, vis.mean(0).round(3).tolist()))}

    for tag, (r, v) in {"clean": (rgb, vis), "stress": occlude(rgb, vis, fps)}.items():
        pos = region_pos(r, fps)
        methods = {n: (lambda s, e, i=i: dominant_bpm(pos[i, s:e], fps)) for i, n in enumerate(NAMES)}
        methods["fusion"] = lambda s, e: dominant_bpm(fuse(pos[:, s:e], v.T[:, s:e], fps)[0], fps)
        metrics, rows = score(methods, lambda s, e: dominant_bpm(ref[s:e], fps), fps, len(r),
                              round(a.window * fps), round(a.stride * fps))
        report[tag] = {"metrics": metrics, "windows": rows}
        print(tag, json.dumps(metrics, indent=1))
    report["note"] = ("stress = synthetic region loss; overlapping windows, single recording, "
                      "not independent evidence; research benchmark only.")
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
