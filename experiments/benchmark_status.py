"""Benchmark-integrity helpers (pure Python, no heavy dependencies so they run in the lightweight test workflow).

Rules enforced here:
  * Missing SmartSpectra output is never a number. It is reported as INVALID_NO_OUTPUT.
  * Zero-confidence / non-positive samples are not readings (SmartSpectra documents that confidence is 0 until the
    metric's analysis window has filled: 12 s for pulse rate) and are excluded from scoring, but counted and reported.
  * A baseline with no scored windows is invalid, whatever the reason.
"""
import math

INVALID_NO_OUTPUT = "INVALID_NO_OUTPUT"
INVALID_NO_ALIGNED_WINDOWS = "INVALID_NO_ALIGNED_WINDOWS"
VALID = "VALID"


def _num(x):
    """Return a float for plain numbers and for protobufjs/long.js Long dicts ({low, high, unsigned}); else None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    if isinstance(x, dict) and "low" in x and "high" in x:
        low, high = int(x["low"]) & 0xFFFFFFFF, int(x["high"])
        if not x.get("unsigned", False) and high >= 0x80000000:
            high -= 0x100000000
        return float(high * 4294967296 + low)
    return None


def usable_sample(value, confidence):
    """A pulse sample is a reading only if it is a positive number and its confidence is not exactly zero."""
    v = _num(value)
    if v is None or v <= 0:
        return False
    c = _num(confidence) if confidence is not None else None
    return c is None or c > 0


def split_samples(raw):
    """raw: iterable of (time_s, value, confidence_or_None). Returns (usable [(t, v)], discarded_count)."""
    usable, dropped = [], 0
    for t, v, c in raw:
        if usable_sample(v, c) and _num(t) is not None:
            usable.append((float(_num(t)), float(_num(v))))
        else:
            dropped += 1
    return usable, dropped


def arm_stats(rows, key, ref_key="reference_bpm"):
    valid = [r for r in rows if r[key] is not None]
    err = [r[key] - r[ref_key] for r in valid]
    n = len(err)
    return {
        "windows_scored": n,
        "coverage_pct": 100 * n / max(1, len(rows)),
        "mae_bpm": sum(abs(e) for e in err) / n if n else None,
        "rmse_bpm": math.sqrt(sum(e * e for e in err) / n) if n else None,
        "within_5_bpm_pct_of_all_windows": 100 * sum(abs(e) <= 5 for e in err) / max(1, len(rows)),
    }


def smartspectra_arm(rows, usable_count, discarded_count=0):
    """Statistics for the SmartSpectra arm, or an explicit invalid status. Never fabricates a score."""
    base = {"usable_samples": usable_count, "discarded_zero_confidence_or_nonpositive": discarded_count}
    if usable_count == 0:
        return {**base, "status": INVALID_NO_OUTPUT, "windows_scored": 0, "coverage_pct": 0.0, "mae_bpm": None,
                "rmse_bpm": None, "within_5_bpm_pct_of_all_windows": None,
                "note": "SmartSpectra produced no usable pulse-rate samples; no baseline exists for this recording."}
    stats = arm_stats(rows, "smartspectra_bpm")
    if stats["windows_scored"] == 0:
        return {**base, **stats, "status": INVALID_NO_ALIGNED_WINDOWS, "mae_bpm": None, "rmse_bpm": None,
                "within_5_bpm_pct_of_all_windows": None,
                "note": "SmartSpectra produced samples but none fall inside a scored reference window."}
    return {**base, **stats, "status": VALID}


def gate_exit_code(report):
    """0 only when the SmartSpectra baseline is valid. Used as the final CI gate (after artifacts are uploaded)."""
    status = report.get("arms", {}).get("SmartSpectra", {}).get("status")
    return 0 if status == VALID else 1
