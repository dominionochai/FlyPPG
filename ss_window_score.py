"""Timestamp-aligned scoring for SmartSpectra logs.

Reference CSV columns: video,start_s,end_s,hr_bpm
SDK logs: one <video>.txt per arm, containing timestamped edge-metrics events.
Timestamps are interpreted as elapsed microseconds from video start unless an
explicit --timestamp-offset-us is supplied. No SDK calls are made here.
"""
import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

EVENT_RE = re.compile(r"Got edge metrics at (\d+) microseconds: (\{.*\})")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _timestamp(value, fallback):
    if _finite_number(value):
        return int(value)
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return fallback


def readings(log_text):
    samples = []

    def walk(node, callback_us):
        if isinstance(node, dict):
            for key, value in node.items():
                if re.sub(r"[^a-z]", "", key.lower()) == "pulserate":
                    for item in value if isinstance(value, list) else [value]:
                        if _finite_number(item):
                            samples.append({"value": float(item), "timestamp_us": callback_us,
                                            "confidence": None, "stable": None})
                        elif isinstance(item, dict):
                            value = next((item[k] for k in ("value", "rate", "bpm", "pulse_rate", "pulseRate")
                                          if k in item and _finite_number(item[k])), None)
                            if value is None:
                                continue
                            stamp = next((item[k] for k in ("timestamp", "time_stamp", "timeStamp")
                                          if k in item), None)
                            confidence = item.get("confidence")
                            stable = item.get("stable")
                            samples.append({
                                "value": float(value),
                                "timestamp_us": _timestamp(stamp, callback_us),
                                "confidence": float(confidence) if _finite_number(confidence) else None,
                                "stable": stable if isinstance(stable, bool) else None,
                            })
                else:
                    walk(value, callback_us)
        elif isinstance(node, list):
            for item in node:
                walk(item, callback_us)

    for match in EVENT_RE.finditer(log_text):
        callback_us = int(match.group(1))
        try:
            walk(json.loads(match.group(2)), callback_us)
        except (json.JSONDecodeError, TypeError):
            continue
    return sorted(samples, key=lambda sample: sample["timestamp_us"])


def load_reference(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video", "start_s", "end_s", "hr_bpm"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Reference CSV must contain video,start_s,end_s,hr_bpm")
    grouped = defaultdict(list)
    for row in rows:
        video = row["video"].strip()
        start, end, hr = float(row["start_s"]), float(row["end_s"]), float(row["hr_bpm"])
        if not video or not all(math.isfinite(x) for x in (start, end, hr)):
            raise ValueError(f"Invalid reference row: {row}")
        if start < 0 or end <= start or not 30 <= hr <= 240:
            raise ValueError(f"Out-of-range reference row: {row}")
        grouped[video].append({"start_s": start, "end_s": end, "hr_bpm": hr})
    for video, windows in grouped.items():
        windows.sort(key=lambda row: row["start_s"])
        for previous, current in zip(windows, windows[1:]):
            if current["start_s"] < previous["end_s"]:
                raise ValueError(f"Overlapping reference windows for {video}")
    return dict(grouped)


def score_window(samples, start_s, end_s, timestamp_offset_us=0,
                 min_confidence=None, require_stable=False):
    start_us = int(round(start_s * 1_000_000)) + timestamp_offset_us
    end_us = int(round(end_s * 1_000_000)) + timestamp_offset_us
    values = []
    for sample in samples:
        stamp = sample["timestamp_us"]
        if not start_us <= stamp < end_us:
            continue
        if min_confidence is not None:
            if sample["confidence"] is None or sample["confidence"] < min_confidence:
                continue
        if require_stable and sample["stable"] is not True:
            continue
        values.append(sample["value"])
    return float(np.median(values)) if values else None


def _paired_video_delta(arm_rows, control_rows, rng, iterations):
    arm_by_video, control_by_video = defaultdict(list), defaultdict(list)
    for row in arm_rows:
        if row["estimate_bpm"] is not None:
            arm_by_video[row["video"]].append(abs(row["estimate_bpm"] - row["hr_bpm"]))
    for row in control_rows:
        if row["estimate_bpm"] is not None:
            control_by_video[row["video"]].append(abs(row["estimate_bpm"] - row["hr_bpm"]))
    common = sorted(set(arm_by_video) & set(control_by_video))
    if not common:
        return None
    deltas = np.array([np.mean(arm_by_video[v]) - np.mean(control_by_video[v]) for v in common])
    draws = rng.choice(deltas, size=(iterations, len(deltas)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "paired_videos": len(common),
        "delta_mae_bpm": float(deltas.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "interpretation": "improvement" if high < 0 else "regression" if low > 0 else "inconclusive",
    }


def evaluate(output_dir, references, control="control", timestamp_offset_us=0,
             min_confidence=None, require_stable=False, bootstrap=5000, seed=0):
    arms = sorted(name for name in os.listdir(output_dir)
                  if os.path.isdir(os.path.join(output_dir, name)))
    per_arm, summary = {}, {}
    for arm in arms:
        rows = []
        for video, windows in references.items():
            log_path = os.path.join(output_dir, arm, video + ".txt")
            samples = readings(open(log_path, encoding="utf-8", errors="replace").read()) if os.path.isfile(log_path) else []
            for window in windows:
                estimate = score_window(samples, window["start_s"], window["end_s"],
                                        timestamp_offset_us, min_confidence, require_stable)
                rows.append({
                    "video": video, "start_s": window["start_s"], "end_s": window["end_s"],
                    "hr_bpm": window["hr_bpm"], "estimate_bpm": estimate,
                    "absolute_error_bpm": abs(estimate - window["hr_bpm"]) if estimate is not None else None,
                })
        per_arm[arm] = rows
        valid = [row for row in rows if row["estimate_bpm"] is not None]
        errors = np.array([row["estimate_bpm"] - row["hr_bpm"] for row in valid], dtype=float)
        summary[arm] = {
            "reference_windows": len(rows),
            "valid_windows": len(valid),
            "coverage_pct": 100.0 * len(valid) / max(1, len(rows)),
            "mae_bpm": float(np.mean(np.abs(errors))) if len(errors) else None,
            "rmse_bpm": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            "within_5_bpm_pct_all_windows": 100.0 * sum(abs(x) <= 5 for x in errors) / max(1, len(rows)),
        }
    rng = np.random.default_rng(seed)
    paired = {}
    if control in per_arm:
        for arm in arms:
            if arm != control:
                paired[arm] = _paired_video_delta(per_arm[arm], per_arm[control], rng, bootstrap)
    return {
        "protocol": {
            "reference_type": "timestamped reference HR windows",
            "sdk_timestamp_unit": "microseconds",
            "timestamp_offset_us": timestamp_offset_us,
            "window_policy": "median of SDK samples whose timestamps fall within each reference interval",
            "bootstrap_unit": "video (mean window absolute error per video)",
            "control_arm": control,
            "warning": "Confirm SDK timestamps share the video clock and set --timestamp-offset-us if required.",
        },
        "arms": summary,
        "paired_vs_control": paired,
        "window_results": per_arm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Directory containing one subdirectory per arm")
    parser.add_argument("--ref", required=True, help="CSV: video,start_s,end_s,hr_bpm")
    parser.add_argument("--control", default="control")
    parser.add_argument("--timestamp-offset-us", type=int, default=0)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--require-stable", action="store_true")
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()
    if args.boot < 1:
        parser.error("--boot must be positive")
    result = evaluate(args.out, load_reference(args.ref), args.control,
                      args.timestamp_offset_us, args.min_confidence,
                      args.require_stable, args.boot, args.seed)
    print("Arm              Coverage     MAE    RMSE   <=5 bpm")
    for arm, row in result["arms"].items():
        mae = "n/a" if row["mae_bpm"] is None else f"{row['mae_bpm']:.2f}"
        rmse = "n/a" if row["rmse_bpm"] is None else f"{row['rmse_bpm']:.2f}"
        print(f"{arm:16} {row['coverage_pct']:7.1f}% {mae:>7} {rmse:>7} "
              f"{row['within_5_bpm_pct_all_windows']:8.1f}%")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
