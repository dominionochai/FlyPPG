"""Final CI gate: exit non-zero unless the SmartSpectra baseline is valid. Run AFTER artifacts are uploaded."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_status as status

def main(path):
    p = Path(path)
    if not p.exists():
        print(f"GATE FAIL: {path} was not produced (scoring step did not complete)."); return 1
    report = json.loads(p.read_text()); arm = report.get("arms", {}).get("SmartSpectra", {})
    lines = ["## Benchmark status", f"- SmartSpectra baseline: **{arm.get('status', 'MISSING')}**",
             f"- usable pulse samples: {arm.get('usable_samples')}, discarded zero-confidence/non-positive: {arm.get('discarded_zero_confidence_or_nonpositive')}"]
    for name in ("raw_POS", "FlyPPG_v3"):
        a = report["arms"].get(name, {}); lines.append(f"- {name}: windows scored {a.get('windows_scored')}, MAE {a.get('mae_bpm')}")
    if arm.get("note"): lines.append(f"- note: {arm['note']}")
    sdk = report.get("smartspectra_sdk_summary") or {}
    if sdk: lines.append(f"- SDK: packets={sdk.get('packet_count')}, pulse packets={sdk.get('pulse_packet_count')}, validation events={sdk.get('validation_event_count')}")
    text = "\n".join(lines); print(text)
    summary = __import__("os").environ.get("GITHUB_STEP_SUMMARY")
    if summary: open(summary, "a").write(text + "\n")
    code = status.gate_exit_code(report)
    if code: print("GATE FAIL: no valid SmartSpectra baseline. This is a failed experiment, not a score of zero.")
    return code

if __name__ == "__main__": sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "head_to_head.json"))
