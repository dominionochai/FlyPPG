"""Score SmartSpectra pulse output against a reference PPG, per 10 s window.
Usage: python score_ss.py RUNS_DIR REFERENCE.txt VIDEO   (RUNS_DIR/<name>_rN/metrics.json)
SmartSpectra's estimate for a window = median of its confident readings inside it; windows with none count as missed."""
import json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landmark_regions import dominant_bpm

runs, ref_path, video = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
ref = np.loadtxt(ref_path)[:, 0]
win, step = round(10 * fps), round(5 * fps)
starts = range(0, len(ref) - win + 1, step)
truth = {a: dominant_bpm(ref[a:a + win], fps) for a in starts}


def score(path):
    d = json.loads(Path(path).read_text())
    r = [(int(s.get("timestamp", 0)) / 1e6, s.get("value", 0), s.get("confidence", 0))
         for p in d["packets"] for s in p.get("metrics", {}).get("cardio", {}).get("pulseRate", [])]
    t, v = (np.array(x) for x in zip(*[(a, b) for a, b, c in r if c > 0])) if any(c > 0 for *_, c in r) else (np.array([]), np.array([]))
    errs = [abs(np.median(v[(t >= a / fps) & (t < (a + win) / fps)]) - truth[a])
            for a in starts if len(t) and ((t >= a / fps) & (t < (a + win) / fps)).any()]
    return {"samples": len(v), "scored": len(errs), "mae": float(np.mean(errs)) if errs else None,
            "within5": float(np.mean(np.array(errs) <= 5) * 100) if errs else None}


groups, rows = defaultdict(list), {}
for m in sorted(runs.glob("*/metrics.json")):
    rows[m.parent.name] = score(m)
    groups[m.parent.name.rsplit("_r", 1)[0]].append(rows[m.parent.name])
print(f"windows total: {len(starts)} | reference window pulse range {min(truth.values()):.0f}-{max(truth.values()):.0f} BPM\n")
print("| arm | runs | windows scored (mean) | MAE BPM (mean of runs) | MAE per run | within 5 BPM % (mean) |\n|---|---|---|---|---|---|")
for g, rs in groups.items():
    ok = [r["mae"] for r in rs if r["mae"] is not None]
    print(f"| {g} | {len(rs)} | {np.mean([r['scored'] for r in rs]):.1f} | {np.mean(ok):.1f} | " if ok else f"| {g} | {len(rs)} | 0 | none | ", end="")
    print((", ".join(f"{x:.1f}" for x in ok) + f" | {np.mean([r['within5'] for r in rs if r['within5'] is not None]):.0f} |") if ok else "- | - |")
Path(runs, "scores.json").write_text(json.dumps(rows, indent=1))
