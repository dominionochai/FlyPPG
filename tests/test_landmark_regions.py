import unittest

import numpy as np

from landmark_regions import cheek_visibility, dominant_bpm, fill_gaps, fuse, snr


def pulse(bpm, fps=30, n=300, noise=0.0, seed=0):
    t = np.arange(n) / fps
    return np.sin(2 * np.pi * bpm / 60 * t) + noise * np.random.default_rng(seed).normal(size=n)


class LandmarkRegionTests(unittest.TestCase):
    def test_frontal_face_sees_both_cheeks(self):
        self.assertEqual(cheek_visibility(0, 10, 5), (1.0, 1.0))

    def test_turn_hides_receding_cheek_only(self):
        left, right = cheek_visibility(0, 10, 1)
        self.assertLess(left, 0.1)
        self.assertEqual(right, 1.0)

    def test_fill_gaps_interpolates_and_keeps_length(self):
        x = np.arange(10, dtype=float).reshape(10, 1, 1).repeat(3, 2)
        x[3:6] = np.nan
        np.testing.assert_allclose(fill_gaps(x)[:, 0, 0], np.arange(10))

    def test_snr_prefers_clean_pulse(self):
        self.assertGreater(snr(pulse(72), 30), snr(pulse(72, noise=3), 30))

    def test_fusion_ignores_hidden_noise_region(self):
        clean, junk = pulse(72), pulse(110, noise=2, seed=1)
        vis = np.vstack([np.ones(300), np.zeros(300)])
        fused, w = fuse(np.vstack([clean, junk]), vis, 30)
        self.assertAlmostEqual(dominant_bpm(fused, 30), 72, delta=2)
        self.assertEqual(w[1], 0.0)


if __name__ == "__main__":
    unittest.main()
