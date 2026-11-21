"""Summarize the budgeted sweep: per clip, SmartSpectra alone (orig) vs with the framing adapter (pad40), scored vs reference.
Usage: python sweep_report.py OUT_DIR   (OUT_DIR/runs/<clip>_<variant>/{summary,metrics}.json, OUT_DIR/refs/<clip>.txt + .fps)"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from landmark_regions import dominant_bpm

out = Path(sys.argv[1])


def readings(run):
    d = json.loads((run / "metrics.json").read_text())
    r = [(int(s.get("timestamp", 0)) / 1e6, s.get("value", 0), s.get("confidence", 0))
         for p in d["packets"] for s in p.get("metrics", {}).get("cardio", {}).get("pulseRate", [])]
    return [x for x in r if x[2] > 0]


def mae(run, clip):
    ref = np.loadtxt(out / "refs" / f"{clip}.txt")[:, 0]
    fps = float((out / "refs" / f"{clip}.fps").read_text())
    r = readings(run)
    if not r:
        return None
    t, v = np.array([x[0] for x in r]), np.array([x[1] for x in r])
    win, step, errs = round(10 * fps), round(5 * fps), []
    for a in range(0, len(ref) - win + 1, step):
        m = (t >= a / fps) & (t < (a + win) / fps)
        if m.any():
            errs.append(abs(np.median(v[m]) - dominant_bpm(ref[a:a + win], fps)))
    return (float(np.mean(errs)), float(np.mean(np.array(errs) <= 5) * 100), len(errs)) if errs else None


def row(run, clip):
    if not (run / "summary.json").exists():
        return "not run", "-", "-"
    s = json.loads((run / "summary.json").read_text())
    m = mae(run, clip)
    return s["status"], s["pulse_samples_nonzero_confidence"], (f"{m[0]:.1f} BPM ({m[1]:.0f}% within 5, {m[2]} windows)" if m else "-")


runs = out / "runs"
clips = sorted({p.name.split("_")[0] for p in runs.glob("c*_*")})
print("| clip | alone: status | alone: readings | with adapter: status | with adapter: readings | adapter error vs reference |\n|---|---|---|---|---|---|")
for c in clips:
    a, b = row(runs / f"{c}_orig", c), row(runs / f"{c}_pad40", c)
    print(f"| {c} | {a[0]} | {a[1]} | {b[0]} | {b[1]} | {b[2]} |")
print("\nFraming threshold sweep (same clip, canvas extended by pad% of frame height):\n\n| extension | status | readings | error vs reference |\n|---|---|---|---|")
for p in sorted(runs.glob("thr_pad*"), key=lambda p: int(p.name[7:])):
    s = row(p, "c01")
    print(f"| {p.name[7:]}% | {s[0]} | {s[1]} | {s[2]} |")
if (out / "spent.txt").exists():
    print(f"\nSDK video seconds sent: {(out / 'spent.txt').read_text().split()[-1]}")
