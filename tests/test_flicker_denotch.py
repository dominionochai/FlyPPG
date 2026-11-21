import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import flicker_experiment as fe


class BandpassSeriesTests(unittest.TestCase):
    def test_band_is_zero_phase_and_idempotent(self):
        fps, n = 30.0, 1800
        t = np.arange(n) / fps
        x = np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 2.9 * t)
        b1 = fe.bandpass_series(x, fps)
        b2 = fe.bandpass_series(b1, fps)
        self.assertGreater(float(np.corrcoef(b1, b2)[0, 1]), 0.99)
        # In-band content must NOT be phase-flipped (a Gaussian-difference bandpass does that above its mid cutoff).
        self.assertGreater(float(np.corrcoef(b1, x)[0, 1]), 0.99)

    def test_out_of_band_content_removed(self):
        fps, n = 30.0, 1800
        t = np.arange(n) / fps
        x = 0.5 + 0.3 * np.sin(2 * np.pi * 0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 1.5 * t)
        y = fe.bandpass_series(x, fps)
        self.assertGreater(0.1, float(np.corrcoef(y, np.sin(2 * np.pi * 0.05 * t))[0, 1]) ** 2)
        self.assertGreater(float(np.corrcoef(y, np.sin(2 * np.pi * 1.5 * t))[0, 1]), 0.99)


class DominantFreqTests(unittest.TestCase):
    def test_single_tone_detected_with_high_concentration(self):
        fps, n = 30.0, 1800
        t = np.arange(n) / fps
        freq, conc = fe.dominant_freq(np.sin(2 * np.pi * 1.4667 * t), fps)
        self.assertAlmostEqual(freq, 1.4667, delta=0.02)
        self.assertGreater(conc, 0.8)

    def test_constant_series_has_no_peak(self):
        freq, conc = fe.dominant_freq(np.ones(300), 30.0)
        self.assertEqual(freq, 0.0)


class DenotchGainsTests(unittest.TestCase):
    def series_video(self, t, per_channel_gain):
        return 0.5 * np.stack([1 + w * t for w in per_channel_gain], 1)

    def test_clean_pulse_level_series_is_exact_noop(self):
        fps, n = 30.0, 1800
        t = np.arange(n) / fps
        # Pulse-level series: below PULSEAMP even with concentration high.
        clean = 0.5 + 0.002 * np.sin(2 * np.pi * 1.2 * t)
        gain, info = fe.denotch_gains(np.stack([clean] * 3, 1), fps)
        self.assertFalse(info["armed"])
        np.testing.assert_allclose(gain, 1.0)

    def test_entrainment_tone_is_detected_and_cancelled(self):
        fps, n = 30.0, 1800
        t = np.arange(n) / fps
        flicker = 0.08 * np.sin(2 * np.pi * 1.4667 * t)  # 88 bpm trap
        m = self.series_video(flicker, (1.4, 0.8, 0.55))
        gain, info = fe.denotch_gains(m, fps)
        self.assertTrue(info["armed"])
        self.assertAlmostEqual(info["freq_hz"], 1.4667, delta=0.05)
        corrected = np.log(m) + np.log(gain.astype(np.float64))
        for c in range(3):
            before = float(fe.bandpass_series(np.log(m[:, c]), fps).std())
            after = float(fe.bandpass_series(corrected[:, c], fps).std())
            self.assertLess(after, 0.02 * before)

    def test_broadband_flicker_is_cancelled(self):
        fps, n = 30.0, 1800
        flicker = fe.flicker_signal(n, fps, 1)
        m = self.series_video(0.12 * flicker, (1.0, 1.0, 1.0))
        gain, info = fe.denotch_gains(m, fps)
        self.assertTrue(info["armed"])
        corrected = np.log(m) + np.log(gain.astype(np.float64))
        for c in range(3):
            before = float(fe.bandpass_series(np.log(m[:, c]), fps).std())
            after = float(fe.bandpass_series(corrected[:, c], fps).std())
            self.assertLess(after, 0.2 * before)

    def test_gain_capped_to_protect_against_overcorrection(self):
        fps, n = 30.0, 1800
        t = np.arange(n) / fps
        flicker = 0.5 * np.sin(2 * np.pi * 1.5 * t)
        gain, info = fe.denotch_gains(self.series_video(flicker, (1.0, 1.0, 1.0)), fps)
        self.assertLessEqual(float(np.max(np.abs(np.log(gain + 1e-12)))), fe.MAX_CORR + 1e-4)

    def test_short_series_is_noop(self):
        gain, info = fe.denotch_gains(np.ones((10, 3)), 30.0)
        self.assertFalse(info["armed"])
        np.testing.assert_allclose(gain, 1.0)


class NearFreqPctTests(unittest.TestCase):
    def test_windows_at_flicker_frequency_counted(self):
        samples = [(t, 88.0) for t in np.arange(0, 60, 0.5)]
        pct = flicker_score_module().near_freq_pct(samples, 88.0, 60.0)
        self.assertEqual(pct, 100.0)

    def test_far_from_flicker_frequency_not_counted(self):
        samples = [(t, 62.0) for t in np.arange(0, 60, 0.5)]
        pct = flicker_score_module().near_freq_pct(samples, 88.0, 60.0)
        self.assertEqual(pct, 0.0)

    def test_no_flicker_bpm_returns_none(self):
        self.assertIsNone(flicker_score_module().near_freq_pct([(0.0, 70.0)], None, 60.0))


def flicker_score_module():
    import flicker_score

    return flicker_score


if __name__ == "__main__":
    unittest.main()
