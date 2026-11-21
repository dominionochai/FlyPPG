"""Landmark-tracked regions: each region alone vs visibility-weighted fusion, on the MCD clip.

--stress simulates turning: spans where one region carries pulse-free noise and reports
zero visibility (what the tracker reports when that cheek turns away). Synthetic; it tests
the fusion mechanism, not the accuracy of the visibility estimate.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landmark_regions import (NAMES, dominant_bpm, fuse, region_pos, track_video,
                              window_local_quality_vote)


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
        win_frames = round(a.window * fps)
        stride_frames = round(a.stride * fps)
        metrics, rows = score(methods, lambda s, e: dominant_bpm(ref[s:e], fps), fps, len(r),
                              win_frames, stride_frames)

        # New candidate: calculate visibility and spectral confidence separately
        # inside each scoring window, then combine regional HR estimates robustly.
        local_rows = window_local_quality_vote(pos, v, fps, win_frames, stride_frames)
        local_errors, best_errors = [], []
        for item in local_rows:
            truth = dominant_bpm(ref[item["start_frame"]:item["stop_frame"]], fps)
            item["reference_bpm"] = truth
            local_errors.append(abs(item["bpm"] - truth) if np.isfinite(item["bpm"]) else np.nan)
            best_errors.append(abs(item["best_weight_bpm"] - truth)
                               if np.isfinite(item["best_weight_bpm"]) else np.nan)
        local_errors = np.asarray(local_errors, dtype=float)
        best_errors = np.asarray(best_errors, dtype=float)

        def summarize(errors):
            valid = errors[np.isfinite(errors)]
            return {
                "windows_scored": int(len(valid)),
                "coverage_pct": float(100 * len(valid) / max(1, len(errors))),
                "mae_bpm": float(valid.mean()) if len(valid) else None,
                "rmse_bpm": float(np.sqrt(np.mean(valid ** 2))) if len(valid) else None,
                "within5_pct": float(np.mean(valid <= 5) * 100) if len(valid) else None,
            }

        metrics["window_local_quality_vote"] = summarize(local_errors)
        metrics["window_best_weight_region_diagnostic"] = summarize(best_errors)
        report[tag] = {"metrics": metrics, "windows": rows,
                       "window_local_quality_vote": local_rows}
        print(tag, json.dumps(metrics, indent=1))
    report["note"] = ("stress = synthetic region loss; overlapping windows, single recording, "
                      "not independent evidence; research benchmark only.")
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
