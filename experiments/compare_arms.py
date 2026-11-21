"""Compare SmartSpectra runs of the same clip under different FlyPPG arms: usable-pulse coverage, confidence, stability.
No reference pulse here, so this measures how much usable signal SmartSpectra returns, not BPM accuracy.
Usage: python compare_arms.py RUNS_DIR   (RUNS_DIR/<arm>/summary.json)"""
import json, sys
from pathlib import Path
import numpy as np

rows = []
for p in sorted(Path(sys.argv[1]).glob("*/summary.json")):
    s = json.loads(p.read_text())
    v = s.get("validation_summary") or []
    ok = sum(e["count"] for e in v if " kOk" in e["message"])
    total = sum(e["count"] for e in v) or 1
    smp = [x for x in s.get("pulse_rate_samples", []) if x.get("confidence", 0) > 0]
    rate = np.array([x.get("value", x.get("rate", np.nan)) for x in smp], float)
    conf = np.array([x["confidence"] for x in smp], float)
    rows.append({"arm": p.parent.name, "pulse_samples": len(smp), "ok_frames_pct": round(100 * ok / total, 1),
                 "mean_conf": round(float(conf.mean()), 3) if len(conf) else None,
                 "rate_mean": round(float(np.nanmean(rate)), 1) if len(rate) else None,
                 "rate_std": round(float(np.nanstd(rate)), 2) if len(rate) else None, "errors": s.get("error_count")})
print("| arm | pulse samples | frames passing chest/face gate % | mean confidence | rate mean | rate std | errors |\n|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['arm']} | {r['pulse_samples']} | {r['ok_frames_pct']} | {r['mean_conf']} | {r['rate_mean']} | {r['rate_std']} | {r['errors']} |")
Path(sys.argv[1], "compare.json").write_text(json.dumps(rows, indent=1))
