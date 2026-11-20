#!/usr/bin/env python3
"""Join SmartSpectra runs with measured video framing and emit a rule-based, evidence-citing verdict.
Usage: summarize_diagnosis.py OUTDIR   (expects OUTDIR/runs/<name>/summary.json and OUTDIR/diag/<video>/{report.json,face_geometry.csv})"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

CHEST = 7  # ValidationCode value for "chest not visible" in the installed SDK (js/constants.js); verified against validation messages below


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def geometry(path):
    rows = []
    if Path(path).exists():
        for r in csv.DictReader(open(path)):
            rows.append({k: (float(v) if v not in ("", "True", "False") else (v == "True" if v else None)) if k not in ("face",) else v == "True" for k, v in r.items()})
    return rows


def join(metrics, rows):
    """For each SDK validation event, attach the nearest sampled frame's framing (below-chin margin in face heights)."""
    pts = [(r["t_s"], r.get("below_chin_face_heights")) for r in rows if r.get("face") and r.get("below_chin_face_heights") is not None]
    if not pts or not metrics:
        return None
    ts = np.array([p[0] for p in pts]); bc = np.array([p[1] for p in pts]); out = []
    for v in metrics.get("validations", []):
        t = (v.get("timestamp_us") or 0) / 1e6
        i = int(np.abs(ts - t).argmin())
        if abs(ts[i] - t) <= 2.0:
            out.append((v["code"], bc[i]))
    return out


def main(root):
    root = Path(root)
    runs = {}
    for d in sorted((root / "runs").glob("*")):
        s = load(d / "summary.json")
        if s:
            runs[d.name] = {"summary": s, "metrics": load(d / "metrics.json")}
    diags = {d.name: {"report": load(d / "report.json"), "rows": geometry(d / "face_geometry.csv")} for d in sorted((root / "diag").glob("*"))}
    table = []
    for name, r in runs.items():
        s = r["summary"]
        chest = sum(x["count"] for x in s["validation_summary"] if x["message"].startswith(f"{CHEST} ") or "chest" in x["message"].lower())
        table.append({"run": name, "video": s["video"], "requested": s["requested_metrics"], "status": s["status"], "packets": s["packet_count"],
                      "pulse_packets": s["pulse_packet_count"], "pulse_samples_nonzero_conf": s["pulse_samples_nonzero_confidence"],
                      "breathing_packets": s["breathing_packet_count"], "validation_events": s["validation_event_count"], "chest_events": chest,
                      "errors": s["error_count"], "top_validation": s["validation_summary"][:4], "transitions": [t["status"] for t in s["status_transitions"]]})
    framing = {}
    for name, d in diags.items():
        rep = d["report"] or {}
        framing[name] = {"detection_rate": rep.get("face_detection_rate"), "below_chin_face_heights": (rep.get("framing_summary") or {}).get("below_chin_face_heights"),
                         "face_h_frac": (rep.get("framing_summary") or {}).get("face_h_frac"), "size": [rep.get("width"), rep.get("height")],
                         "decoded_frames": rep.get("decoded_frame_count"), "timestamps": rep.get("timestamps"), "flagged": rep.get("flagged_discontinuity_frames"),
                         "decode_error_lines": (rep.get("ffmpeg_decode_errors") or {}).get("stderr_line_count"), "conversion_check": rep.get("conversion_check")}
    joined = {}
    for name, r in runs.items():
        key = "control" if name.startswith("control") else "mcd"
        j = join(r["metrics"], diags.get(key, {}).get("rows", []))
        if j:
            arr = np.array(j); chest = arr[:, 0] == CHEST
            joined[name] = {"events_joined": len(j), "chest_fraction": float(chest.mean()), "mean_below_chin_when_chest": float(arr[chest, 1].mean()) if chest.any() else None,
                            "mean_below_chin_when_other": float(arr[~chest, 1].mean()) if (~chest).any() else None}
    control = next((t for t in table if t["run"].startswith("control")), None); mcd = [t for t in table if t["run"].startswith("mcd")]
    if control is None or not mcd:
        verdict = "INCONCLUSIVE: missing control or MCD run"
    elif control["pulse_samples_nonzero_conf"] > 0 and all(m["pulse_samples_nonzero_conf"] == 0 for m in mcd):
        verdict = "OUTCOME_B: the identical runner/SDK/runtime produces usable pulse on the control clip but not on the MCD recordings -> the MCD recording (content/framing), not the runner, prevents output."
    elif control["pulse_samples_nonzero_conf"] == 0:
        verdict = "OUTCOME_C_SUSPECTED: the control clip also yields no usable pulse in this runner -> runner/SDK/runtime defect suspected; MCD incompatibility cannot be concluded."
    elif any(m["pulse_samples_nonzero_conf"] > 0 for m in mcd):
        verdict = "OUTCOME_A: SmartSpectra produced usable pulse on the MCD recording; score it against the reference."
    else:
        verdict = "INCONCLUSIVE"
    result = {"verdict": verdict, "runs": table, "framing": framing, "validation_vs_framing": joined}
    (root / "diagnosis.json").write_text(json.dumps(result, indent=2))
    md = ["# SmartSpectra MCD diagnosis", "", f"**Verdict:** {verdict}", "", "| run | requested | status | packets | pulse pkts | usable pulse samples | validation events | chest events | errors |", "|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        md.append(f"| {t['run']} | {t['requested']} | {t['status']} | {t['packets']} | {t['pulse_packets']} | {t['pulse_samples_nonzero_conf']} | {t['validation_events']} | {t['chest_events']} | {t['errors']} |")
    md += ["", "## Framing (below-chin space in face heights; detection rate)"]
    for n, f in framing.items():
        bc = f["below_chin_face_heights"] or {}
        md.append(f"- **{n}**: {f['size']}, frames {f['decoded_frames']}, face detected {f['detection_rate']}, below-chin p5/p50/p95 = {bc.get('p5')}/{bc.get('p50')}/{bc.get('p95')}, flagged discontinuities {f['flagged']}")
    md += ["", "## Validation vs framing", json.dumps(joined, indent=1)]
    (root / "diagnosis.md").write_text("\n".join(md))
    print("\n".join(md))
    for t in table:
        print(f"::notice title=SS run {t['run']}::status={t['status']} packets={t['packets']} pulse_pkts={t['pulse_packets']} usable_pulse={t['pulse_samples_nonzero_conf']} validation={t['validation_events']} chest={t['chest_events']} errors={t['errors']} top={t['top_validation'][:1]}")
    for n, f in framing.items():
        print(f"::notice title=framing {n}::detect={f['detection_rate']} below_chin={f['below_chin_face_heights']} size={f['size']} frames={f['decoded_frames']} flagged={f['flagged']} ts={ {k: (f['timestamps'] or {}).get(k) for k in ('implied_fps','interval_std_s','intervals_gt_1p5x_median')} }")
    print(f"::notice title=VERDICT::{verdict}")
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        open(sm, "a").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "diagnostics_out")
