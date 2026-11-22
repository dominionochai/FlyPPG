#!/usr/bin/env python3
"""Score the flicker experiment. Usage: flicker_score.py RUNS_DIR [--flicker-bpm B]  (RUNS_DIR/<condition>_r<N>/{summary,metrics}.json)
Pseudo-reference = SmartSpectra's OWN readings on the clean clip (no ground-truth pulse exists for this dataset), pooled over clean repeats.
Noise floor = disagreement between clean repeats. A condition only counts as changed/improved if it exceeds that floor.
With --flicker-bpm B (entrainment traps) each condition also reports the fraction of windows whose median reading sits within +-2.5 BPM of
B: a reading AT the injected flicker frequency proves the estimate locked onto the light, not the heart — failure by definition."""
import json, sys, argparse, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_status as bs

WIN, STEP = 10.0, 5.0
ORDER = ["clean", "clean_nr", "fl06", "fl06_dn", "fl06_nr", "fl12", "fl12_dn", "fl12_nr", "ent088", "ent088_dn"]
LABEL = {"clean": "clean (control)", "clean_nr": "clean + nearref", "fl06": "6% flicker", "fl06_dn": "6% flicker + denotch", "fl06_nr": "6% flicker + nearref",
         "fl12": "12% flicker", "fl12_dn": "12% flicker + denotch", "fl12_nr": "12% flicker + nearref",
         "ent088": "entrain 88bpm", "ent088_dn": "entrain 88bpm + denotch"}
FREQ_TOL = 2.5   # BPM halo around the injected flicker frequency for the entrainment check


def readings(path):
    d = json.loads(Path(path).read_text()); raw = []
    for p in d.get("packets", []):
        for s in ((p.get("metrics") or {}).get("cardio") or {}).get("pulseRate", []):
            raw.append((bs._num(s.get("timestamp")) / 1e6 if bs._num(s.get("timestamp")) else (bs._num(p.get("timestamp_us")) or 0) / 1e6, s.get("value"), s.get("confidence")))
    return bs.split_samples(raw)[0]


def windows(samples, horizon):
    out = {}
    for a in np.arange(0, horizon - WIN + 1e-6, STEP):
        v = [x for t, x in samples if a <= t < a + WIN]
        out[round(float(a), 1)] = float(np.median(v)) if v else None
    return out


def near_freq_pct(samples, flicker_bpm, horizon):
    """Fraction of window medians within FREQ_TOL of the injected flicker BPM = entrainment evidence."""
    if not flicker_bpm:
        return None
    w = windows(samples, horizon); vals = [v for v in w.values() if v is not None]
    if not vals:
        return 0.0
    return 100.0 * sum(abs(v - flicker_bpm) <= FREQ_TOL for v in vals) / len(vals)


def main(root, flicker_bpm=None):
    root = Path(root); runs = {}
    for d in sorted(root.glob("*_r*")):
        m = re.match(r"(.+)_r(\d+)$", d.name)
        if m and (d / "metrics.json").exists() and (d / "summary.json").exists():
            runs.setdefault(m.group(1), []).append({"samples": readings(d / "metrics.json"), "summary": json.loads((d / "summary.json").read_text())})
    if "clean" not in runs or len(runs["clean"]) < 2:
        print("INCONCLUSIVE: need >=2 clean repeats"); return 1
    horizon = 60.0
    ref_w = {}
    for a in windows(runs["clean"][0]["samples"], horizon):
        vals = [windows(r["samples"], horizon)[a] for r in runs["clean"]]; vals = [v for v in vals if v is not None]; ref_w[a] = float(np.median(vals)) if vals else None
    def mae_vs_ref(samples, ref):
        w = windows(samples, horizon); e = [abs(w[a] - ref[a]) for a in w if w[a] is not None and ref[a] is not None]; return (float(np.mean(e)), len(e)) if e else (None, 0)
    # noise floor: leave-one-out among clean repeats
    floors = []
    for i, r in enumerate(runs["clean"]):
        other = [windows(o["samples"], horizon) for j, o in enumerate(runs["clean"]) if j != i]; ref = {a: (float(np.median([o[a] for o in other if o[a] is not None])) if any(o[a] is not None for o in other) else None) for a in ref_w}
        f, n = mae_vs_ref(r["samples"], ref)
        if f is not None: floors.append(f)
    floor = float(np.mean(floors)) if floors else None
    total_w = len(ref_w); rows = []
    for c in ORDER:
        if c not in runs: continue
        maes, covs, usable, okp, ev, ent = [], [], [], [], [], []
        for r in runs[c]:
            m, n = mae_vs_ref(r["samples"], ref_w); w = windows(r["samples"], horizon)
            if m is not None: maes.append(m)
            covs.append(100 * sum(v is not None for v in w.values()) / total_w); usable.append(len(r["samples"]))
            vs = r["summary"]["validation_summary"]; tot = max(1, r["summary"]["validation_event_count"]); okn = sum(x["count"] for x in vs if x["message"].startswith("0 "))
            okp.append(100 * okn / tot); ev.append(r["summary"]["validation_event_count"])
            e = near_freq_pct(r["samples"], flicker_bpm, horizon)
            if e is not None: ent.append(e)
        rows.append({"condition": LABEL[c], "key": c, "reps": len(runs[c]), "usable_samples": float(np.mean(usable)), "coverage_pct": float(np.mean(covs)),
                     "mae_vs_clean_bpm": float(np.mean(maes)) if maes else None, "gate_ok_pct": float(np.mean(okp)), "validation_events": float(np.mean(ev)),
                     "at_flicker_pct": float(np.mean(ent)) if ent else None})
    hdr = "| condition | reps | usable pulse samples | window coverage % | MAE vs clean (bpm) | at-flicker-freq % | frames passing gate % | validation events |"
    md = [hdr, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mae = "n/a" if r["mae_vs_clean_bpm"] is None else "%.2f" % r["mae_vs_clean_bpm"]
        entc = "n/a" if r["at_flicker_pct"] is None else "%.0f" % r["at_flicker_pct"]
        md.append("| %s | %d | %.0f | %.0f | %s | %s | %.0f | %.0f |" % (r["condition"], r["reps"], r["usable_samples"], r["coverage_pct"], mae, entc, r["gate_ok_pct"], r["validation_events"]))
    floor_txt = "n/a" if floor is None else "%.2f bpm" % floor
    md.append(""); md.append("Noise floor (clean repeat vs clean repeats, leave-one-out): %s. Pseudo-reference = SmartSpectra own clean readings; no ground-truth pulse." % floor_txt)
    verdicts = []
    by = {r["key"]: r for r in rows}
    def arm_verdict(tag, base, arm):
        a, b = by.get(base), by.get(arm)
        if not a or not b or a["mae_vs_clean_bpm"] is None: return f"{tag}: INCONCLUSIVE (flicker arm produced no comparable readings)"
        thr = max(1.0, floor or 0)
        degraded = a["mae_vs_clean_bpm"] > thr or a["coverage_pct"] < by["clean"]["coverage_pct"] - 15
        if b["mae_vs_clean_bpm"] is None: return f"{tag}: FlyPPG arm lost all readings (WORSE)"
        gain = a["mae_vs_clean_bpm"] - b["mae_vs_clean_bpm"]; cov = b["coverage_pct"] - a["coverage_pct"]
        status = "IMPROVED" if (gain >= thr or cov >= 15) and degraded else ("NO_DEGRADATION_TO_FIX" if not degraded else ("NO_CLEAR_DIFFERENCE" if abs(gain) < thr and abs(cov) < 15 else "WORSE"))
        return (f"{tag}: SmartSpectra {'DEGRADED' if degraded else 'NOT degraded'} (MAE {a['mae_vs_clean_bpm']:.2f}, coverage {a['coverage_pct']:.0f}%); "
                f"with FlyPPG MAE {b['mae_vs_clean_bpm']:.2f}, coverage {b['coverage_pct']:.0f}% -> {status} (threshold {thr:.2f} bpm / 15 pts)")
    for s in ("06", "12"):
        for arm, armname in ((f"fl{s}_dn", "denotch"), (f"fl{s}_nr", "nearref")):
            if by.get(arm): verdicts.append(arm_verdict(f"{s}% flicker + {armname}", f"fl{s}", arm))
    if flicker_bpm:
        for key, armname in (("ent088", "raw"), ("ent088_dn", "denotch")):
            r = by.get(key)
            if not r: continue
            entp = r.get("at_flicker_pct")
            state = "ENTRAINED (reading locked to flicker frequency)" if entp is not None and entp >= 50 else ("NOT entrained" if entp is not None else "no comparable readings")
            verdicts.append(f"entrainment trap {flicker_bpm:.0f} bpm ({armname}): {state} — {entp if entp is not None else 0:.0f}% of windows at the flicker frequency, MAE vs clean {r['mae_vs_clean_bpm'] if r['mae_vs_clean_bpm'] is not None else float('nan'):.2f}")
    out = "\n".join(md + ["", "**Verdicts**"] + [f"- {v}" for v in verdicts]); print(out); (root.parent / "SCORES.md").write_text(out)
    print("::notice title=flicker scores::" + " | ".join(verdicts))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs_dir"); ap.add_argument("--flicker-bpm", type=float, default=None)
    a = ap.parse_args(); sys.exit(main(a.runs_dir, a.flicker_bpm))
