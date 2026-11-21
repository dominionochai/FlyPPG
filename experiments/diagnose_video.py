#!/usr/bin/env python3
"""Inspect what a SmartSpectra input video actually contains, instead of trusting container metadata.

Writes to --out:
  report.json         probe data, timestamp regularity, decode errors, frame counts, flagged frames, face-geometry summary
  face_geometry.csv   per-sampled-frame face box and framing measures (joined with SDK validation events later)
  contact_sheet.png   evenly spaced frames with detected face box and the space below the chin (in face heights)
  frames/*.png        start / middle / end frames and frames around any flagged discontinuity
The original dataset is never modified. Usage:
  python diagnose_video.py VIDEO --out DIR [--label NAME] [--reference-video ORIGINAL] [--sample-every 15]
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return 127, "", f"{cmd[0]} not found"
    return p.returncode, p.stdout, p.stderr


def probe(path):
    rc, out, err = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    data = json.loads(out) if rc == 0 and out else {}
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    side = {}
    for sd in v.get("side_data_list", []) or []:
        side.update({k: sd[k] for k in ("rotation",) if k in sd})
    keep = ("codec_name", "width", "height", "sample_aspect_ratio", "display_aspect_ratio", "pix_fmt", "r_frame_rate", "avg_frame_rate",
            "nb_frames", "duration", "time_base", "field_order")
    return {"video": {k: v.get(k) for k in keep}, "tags_rotate": (v.get("tags") or {}).get("rotate"), "side_data": side,
            "format": {k: (data.get("format") or {}).get(k) for k in ("format_name", "duration", "size", "bit_rate")}}


def packet_times(path):
    rc, out, err = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)])
    ts = []
    for line in out.splitlines():
        try:
            ts.append(float(line.strip().split(",")[0]))
        except ValueError:
            pass
    if len(ts) < 3:
        return {"packets": len(ts), "note": "no usable packet timestamps"}
    d = np.diff(np.sort(np.array(ts)))
    med = float(np.median(d))
    return {"packets": len(ts), "median_interval_s": med, "implied_fps": 1 / med if med > 0 else None, "min_interval_s": float(d.min()),
            "max_interval_s": float(d.max()), "interval_std_s": float(d.std()), "intervals_gt_1p5x_median": int((d > 1.5 * med).sum()),
            "intervals_lt_0p5x_median": int((d < 0.5 * med).sum()), "first_pts_s": float(min(ts)), "last_pts_s": float(max(ts))}


def decode_errors(path):
    rc, out, err = run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    lines = [l for l in err.splitlines() if l.strip()]
    return {"exit_code": rc, "stderr_line_count": len(lines), "stderr_head": lines[:20]}


def make_detector():
    try:
        import mediapipe as mp
        det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)

        def f(frame):
            h, w = frame.shape[:2]
            r = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not r.detections:
                return None
            b = max(r.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height).location_data.relative_bounding_box
            return (b.xmin * w, b.ymin * h, b.width * w, b.height * h)
        return f, "mediapipe_face_detection"
    except Exception as e:  # fall back to Haar so the diagnostic still runs
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        def f(frame):
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = casc.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
            return tuple(max(faces, key=lambda b: b[2] * b[3])) if len(faces) else None
        return f, f"haar_cascade (mediapipe unavailable: {type(e).__name__})"


def scorer_haar_check(video, geometry, W, H):
    """Replicates experiments/mcd_head_to_head.py:load_frames crop logic (Haar box persisted, centered fallback, 1.8x crop) and checks
    whether its POS region (centre of the crop) actually lies on the face found by MediaPipe."""
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    empty = det.empty()
    mp_by_frame = {g["frame"]: g["box"] for g in geometry if g.get("box")}
    cap = cv2.VideoCapture(video); box = None; idx = 0; n_det = 0; n_fallback = 0; stats = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = [] if empty else det.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(80, 80))
        if len(faces):
            box = tuple(int(v) for v in max(faces, key=lambda b: b[2] * b[3])); n_det += 1
        if box is None:
            side = int(min(H, W) * 0.72); x = (W - side) // 2; y = (H - side) // 2; bw = bh = side; n_fallback += 1
        else:
            x, y, bw, bh = box
        if idx in mp_by_frame:
            mx, my, mw, mh = mp_by_frame[idx]; cx, cy = x + bw / 2, y + bh / 2   # POS ROI is the central region of the crop around this box
            ix = max(0, min(x + bw, mx + mw) - max(x, mx)); iy = max(0, min(y + bh, my + mh) - max(y, my)); inter = ix * iy
            iou = inter / (bw * bh + mw * mh - inter + 1e-9)
            stats.append({"frame": idx, "haar_detected_this_frame": bool(len(faces)), "iou_vs_mediapipe": iou,
                          "roi_centre_inside_mp_face": bool(mx <= cx <= mx + mw and my <= cy <= my + mh), "dist_px": float(np.hypot(cx - (mx + mw / 2), cy - (my + mh / 2)))})
        idx += 1
    cap.release()
    ok_frac = float(np.mean([s["roi_centre_inside_mp_face"] for s in stats])) if stats else None
    return {"cascade_loaded": not empty, "frames": idx, "frames_with_haar_detection": n_det, "frames_using_centered_fallback": n_fallback,
            "compared_frames": len(stats), "roi_centre_on_mediapipe_face_fraction": ok_frac,
            "median_iou_vs_mediapipe": float(np.median([s["iou_vs_mediapipe"] for s in stats])) if stats else None,
            "median_centre_distance_px": float(np.median([s["dist_px"] for s in stats])) if stats else None,
            "haar_detected_on_compared_frames_fraction": float(np.mean([s["haar_detected_this_frame"] for s in stats])) if stats else None}


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--out", required=True); ap.add_argument("--label", default=None)
    ap.add_argument("--reference-video", default=None); ap.add_argument("--sample-every", type=int, default=15)
    a = ap.parse_args()
    out = Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    label = a.label or Path(a.video).stem
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detect, detector_name = make_detector()
    prev_small, diffs, lums, geometry, keep = None, [], [], [], {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60)).astype(np.float32)
        diffs.append(float(np.abs(small - prev_small).mean()) if prev_small is not None else 0.0)
        lums.append(float(small.mean())); prev_small = small
        if idx % a.sample_every == 0:
            box = detect(frame)
            row = {"frame": idx, "t_s": idx / fps, "face": box is not None, "mean_luma": lums[-1], "sharpness": float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())}
            if box is not None:
                x, y, bw, bh = box
                row.update({"face_h_frac": bh / H, "face_w_frac": bw / W, "cx_offset": (x + bw / 2 - W / 2) / W, "cy_frac": (y + bh / 2) / H,
                            "above_head_face_heights": y / bh, "below_chin_face_heights": (H - (y + bh)) / bh, "box": [x, y, bw, bh]})
            geometry.append(row)
        keep[idx] = None  # placeholder; frames are re-read below to bound memory
        idx += 1
    n_decoded = idx
    cap.release()

    d = np.array(diffs[1:]) if len(diffs) > 1 else np.array([0.0])
    med = float(np.median(d)); mad = float(np.median(np.abs(d - med))) + 1e-6
    flagged = [i + 1 for i, v in enumerate(d) if v > med + 12 * mad and v > 15.0]
    want = {0, n_decoded // 2, max(0, n_decoded - 1)}
    for f_ in flagged[:6]:
        want |= {max(0, f_ - 2), f_ - 1 if f_ > 0 else 0, f_, min(n_decoded - 1, f_ + 1)}
    cap = cv2.VideoCapture(a.video); saved = []
    idx = 0
    while idx <= max(want) and want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            name = f"frames/{label}_f{idx:06d}.png"; cv2.imwrite(str(out / name), frame); saved.append(name)
        idx += 1
    cap.release()

    # contact sheet: 12 evenly spaced sampled frames, face box + below-chin measure
    cs_rows = [g for g in geometry]
    picks = [cs_rows[int(i)] for i in np.linspace(0, len(cs_rows) - 1, min(12, len(cs_rows)))] if cs_rows else []
    cap = cv2.VideoCapture(a.video); tiles = []
    for g in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, g["frame"]); ok, fr = cap.read()
        if not ok:
            continue
        if g.get("box"):
            x, y, bw, bh = [int(v) for v in g["box"]]; cv2.rectangle(fr, (x, y), (x + bw, y + bh), (0, 255, 0), max(2, W // 320))
        t = cv2.resize(fr, (320, int(320 * H / W)))
        txt = f"t={g['t_s']:.0f}s " + (f"below chin {g['below_chin_face_heights']:.2f} fh" if g.get("box") else "NO FACE")
        cv2.putText(t, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA); tiles.append(t)
    cap.release()
    if tiles:
        cols = 4; rows = (len(tiles) + cols - 1) // cols; th, tw = tiles[0].shape[:2]
        sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
        for k, t in enumerate(tiles):
            sheet[(k // cols) * th:(k // cols + 1) * th, (k % cols) * tw:(k % cols + 1) * tw] = t
        cv2.imwrite(str(out / "contact_sheet.png"), sheet)

    with open(out / "face_geometry.csv", "w", newline="") as fh:
        keys = ["frame", "t_s", "face", "mean_luma", "sharpness", "face_h_frac", "face_w_frac", "cx_offset", "cy_frac", "above_head_face_heights", "below_chin_face_heights"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(geometry)
    det = [g for g in geometry if g["face"]]
    summ = {k: {"p5": pct([g[k] for g in det], 5), "p50": pct([g[k] for g in det], 50), "p95": pct([g[k] for g in det], 95)}
            for k in ("face_h_frac", "face_w_frac", "cx_offset", "cy_frac", "above_head_face_heights", "below_chin_face_heights")}
    report = {"label": label, "file": Path(a.video).name, "decoded_frame_count": n_decoded, "cv2_reported_fps": fps, "width": W, "height": H,
              "probe": probe(a.video), "timestamps": packet_times(a.video), "ffmpeg_decode_errors": decode_errors(a.video),
              "luma": {"mean": float(np.mean(lums)), "min": float(np.min(lums)), "max": float(np.max(lums))},
              "frame_diff": {"median": med, "mad": mad, "max": float(d.max())}, "flagged_discontinuity_frames": flagged[:50],
              "saved_frames": saved, "detector": detector_name, "sampled_frames": len(geometry), "faces_detected": len(det),
              "face_detection_rate": len(det) / max(1, len(geometry)), "framing_summary": summ}
    if "mediapipe" in detector_name:
        report["scorer_haar_check"] = scorer_haar_check(a.video, geometry, W, H)
    if a.reference_video:
        ref = cv2.VideoCapture(a.reference_video); rn = int(ref.get(cv2.CAP_PROP_FRAME_COUNT)); rw, rh = int(ref.get(3)), int(ref.get(4))
        ok1, rf = ref.read(); cap = cv2.VideoCapture(a.video); ok2, cf = cap.read(); mae = None
        if ok1 and ok2:
            mae = float(np.abs(cv2.resize(rf, (cf.shape[1], cf.shape[0])).astype(np.float32) - cf.astype(np.float32)).mean())
        report["conversion_check"] = {"reference_size": [rw, rh], "converted_size": [W, H], "same_size": (rw, rh) == (W, H),
                                      "reference_frame_count_container": rn, "converted_decoded_frames": n_decoded, "first_frame_mean_abs_diff_0_255": mae,
                                      "reference_probe": probe(a.reference_video), "reference_timestamps": packet_times(a.reference_video)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("label", "decoded_frame_count", "width", "height", "detector", "face_detection_rate", "framing_summary", "flagged_discontinuity_frames", "timestamps")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
