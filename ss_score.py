"""Score SmartSpectra pulse-rate output against per-video reference HR.

Supports protobuf JSON's lowerCamelCase (cardio.pulseRate) and snake_case
(pulse_rate). Each video is scored independently; missing outputs remain
failures for coverage and are not silently removed from that arm's coverage.
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


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _timestamp(value, fallback):
    # Protobuf JSON serializes int64 fields as decimal strings.
    if _number(value):
        return int(value)
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return fallback


def readings(text):
    """Return pulse samples with value, confidence, stability and timestamps."""
    found = []

    def add_sample(item, callback_us):
        if _number(item):
            found.append({"value": float(item), "confidence": None,
                          "stable": None, "timestamp_us": callback_us})
            return
        if not isinstance(item, dict):
            return
        value = next((item[k] for k in ("value", "rate", "bpm", "pulse_rate", "pulseRate")
                      if k in item and _number(item[k])), None)
        if value is None:
            return
        raw_stamp = next((item[k] for k in ("timestamp", "time_stamp", "timeStamp")
                          if k in item), None)
        stamp = _timestamp(raw_stamp, callback_us)
        confidence = item.get("confidence")
        stable = item.get("stable")
        found.append({
            "value": float(value),
            "confidence": float(confidence) if _number(confidence) else None,
            "stable": stable if isinstance(stable, bool) else None,
            "timestamp_us": int(stamp) if _number(stamp) else callback_us,
        })

    def walk(node, callback_us):
        if isinstance(node, dict):
            for key, value in node.items():
                normalized = re.sub(r"[^a-z]", "", key.lower())
                if normalized == "pulserate":
                    items = value if isinstance(value, list) else [value]
                    for item in items:
                        add_sample(item, callback_us)
                else:
                    walk(value, callback_us)
        elif isinstance(node, list):
            for item in node:
                walk(item, callback_us)

    for match in EVENT_RE.finditer(text):
        callback_us = int(match.group(1))
        try:
            walk(json.loads(match.group(2)), callback_us)
        except (json.JSONDecodeError, TypeError):
            continue
    return found


def estimate(samples, min_confidence=None, require_stable=False):
    usable = []
    for sample in samples:
        if min_confidence is not None:
            if sample["confidence"] is None or sample["confidence"] < min_confidence:
                continue
        if require_stable and sample["stable"] is not True:
            continue
        usable.append(sample["value"])
    if not usable:
        return float("nan")
    # Discard the first half as warm-up, matching the original harness policy.
    return float(np.median(usable[len(usable) // 2:]))


def load_reference(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"video", "hr_bpm"}.issubset(rows[0]):
        raise ValueError("ref CSV must contain video,hr_bpm columns")
    result = {}
    for row in rows:
        video = row["video"].strip()
        if not video:
            continue
        hr = float(row["hr_bpm"])
        if not math.isfinite(hr) or not 30 <= hr <= 240:
            raise ValueError(f"Invalid reference HR for {video}: {hr}")
        if video in result:
            raise ValueError(f"Duplicate reference video: {video}")
        result[video] = hr
    return result


def bootstrap_delta(arm_errors, control_errors, rng, iterations):
    common = sorted(set(arm_errors) & set(control_errors))
    if not common:
        return None
    delta = np.array([abs(arm_errors[v]) - abs(control_errors[v]) for v in common])
    draws = rng.choice(delta, size=(iterations, len(delta)), replace=True).mean(axis=1)
    return {
        "n_paired_videos": len(common),
        "delta_mae_bpm": float(delta.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "interpretation": (
            "improvement" if np.percentile(draws, 97.5) < 0 else
            "regression" if np.percentile(draws, 2.5) > 0 else
            "inconclusive"
        ),
    }


def score(output_dir, reference, control="control", min_confidence=None,
          require_stable=False, bootstrap=5000, seed=0):
    arms = sorted(name for name in os.listdir(output_dir)
                  if os.path.isdir(os.path.join(output_dir, name)))
    estimates, errors, report = {}, {}, {}
    for arm in arms:
        estimates[arm], errors[arm] = {}, {}
        for video, reference_hr in reference.items():
            path = os.path.join(output_dir, arm, video + ".txt")
            samples = readings(open(path, encoding="utf-8", errors="replace").read()) if os.path.isfile(path) else []
            value = estimate(samples, min_confidence, require_stable)
            estimates[arm][video] = value
            if math.isfinite(value):
                errors[arm][video] = value - reference_hr

        valid = errors[arm]
        abs_errors = np.abs(list(valid.values()))
        report[arm] = {
            "reference_videos": len(reference),
            "videos_with_readings": len(valid),
            "coverage_pct": 100.0 * len(valid) / max(1, len(reference)),
            "mae_bpm": float(abs_errors.mean()) if len(abs_errors) else None,
            "rmse_bpm": float(np.sqrt(np.mean(np.square(list(valid.values()))))) if len(valid) else None,
            "within_5_bpm_pct_of_all_videos": (
                100.0 * sum(abs(e) <= 5 for e in valid.values()) / max(1, len(reference))
            ),
        }

    rng = np.random.default_rng(seed)
    paired = {}
    if control in errors:
        for arm in arms:
            if arm != control:
                paired[arm] = bootstrap_delta(errors[arm], errors[control], rng, bootstrap)

    return {
        "protocol": {
            "reference_type": "one constant reference HR per video",
            "warmup_policy": "discard first half of pulse-rate samples, median of remainder",
            "confidence_filter": min_confidence,
            "require_stable": require_stable,
            "bootstrap_unit": "video",
            "bootstrap_iterations": bootstrap,
            "control_arm": control,
            "warning": (
                "This scorer does not time-align changing reference HR within a video. "
                "Use short stationary segments or implement timestamp-aligned reference windows."
            ),
        },
        "arms": report,
        "paired_vs_control": paired,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--control", default="control")
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--require-stable", action="store_true")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    reference = load_reference(args.ref)
    result = score(args.out, reference, args.control, args.min_confidence,
                   args.require_stable, args.boot, args.seed)
    print("Arm             Coverage     MAE    RMSE   <=5 bpm")
    for arm, row in result["arms"].items():
        mae = "n/a" if row["mae_bpm"] is None else f"{row['mae_bpm']:.2f}"
        rmse = "n/a" if row["rmse_bpm"] is None else f"{row['rmse_bpm']:.2f}"
        print(f"{arm:16} {row['coverage_pct']:7.1f}% {mae:>7} {rmse:>7} "
              f"{row['within_5_bpm_pct_of_all_videos']:8.1f}%")
    print("\nPaired MAE difference (negative favours the tested arm):")
    for arm, row in result["paired_vs_control"].items():
        if row is None:
            print(f"{arm:16} no paired videos")
        else:
            print(f"{arm:16} {row['delta_mae_bpm']:+.2f} bpm "
                  f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] "
                  f"{row['interpretation']} (n={row['n_paired_videos']})")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
