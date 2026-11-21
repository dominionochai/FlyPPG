"""Stability layer for SmartSpectra pulse output: confidence-weighted tracking with an outlier gate.

Each reading (t_s, bpm, confidence 0-100) moves the estimate by confidence/(confidence+K) of its
innovation. A reading is rejected if it implies a physiologically implausible jump for the
elapsed time. Stability only: it cannot make a wrong pulse right.
"""
import numpy as np

K = 60.0          # confidence at which a reading gets half weight
MAX_SLEW = 6.0    # plausible BPM change per second
GATE_MIN = 8.0    # BPM tolerance floor right after a reading


def stabilize(readings):
    """readings: iterable of (t_s, bpm, confidence) -> list of (t_s, stabilized_bpm)."""
    readings = [r for r in readings if r[2] > 0]
    est, t_prev, out = None, None, []
    if readings:
        est, t_prev = float(np.median([r[1] for r in readings[:7]])), readings[0][0]  # robust start, not one reading
    for t, v, c in readings:
        if abs(v - est) <= GATE_MIN + MAX_SLEW * (t - t_prev):
            est += c / (c + K) * (v - est)
        else:
            out.append((t, est))   # rejected: hold; the gate keeps widening with time since the last accepted reading
            continue
        t_prev = t
        out.append((t, est))
    return out


def jitter(bpm):
    """Mean absolute change between consecutive readings (BPM)."""
    return float(np.abs(np.diff(bpm)).mean()) if len(bpm) > 1 else float("nan")
