import unittest

import numpy as np

from stabilize import jitter, stabilize


class StabilizeTests(unittest.TestCase):
    def readings(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        return [(float(t), 72 + rng.normal(0, 3), 60.0) for t in range(n)]

    def test_rejects_isolated_spike(self):
        r = self.readings()
        r[30] = (30.0, 140.0, 60.0)
        out = np.array([v for _, v in stabilize(r)])
        self.assertLess(abs(out[30] - 72), 5)

    def test_reduces_jitter_without_shifting_mean(self):
        r = self.readings()
        raw = np.array([x[1] for x in r]); out = np.array([v for _, v in stabilize(r)])
        self.assertLess(jitter(out), jitter(raw))
        self.assertAlmostEqual(out.mean(), raw.mean(), delta=2)

    def test_follows_a_real_change(self):
        r = [(float(t), 70.0 if t < 30 else 100.0, 70.0) for t in range(80)]
        self.assertAlmostEqual(stabilize(r)[-1][1], 100.0, delta=3)

    def test_ignores_zero_confidence(self):
        self.assertEqual(len(stabilize([(0.0, 70.0, 0.0), (1.0, 71.0, 50.0)])), 1)


if __name__ == "__main__":
    unittest.main()
