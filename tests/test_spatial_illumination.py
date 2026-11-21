import unittest
import numpy as np

from ss_preprocess import spatial_illumination_gain


class SpatialIlluminationGainTests(unittest.TestCase):
    def test_output_shape_finite_and_bounded(self):
        frames = np.full((4, 96, 96, 3), 0.4, dtype=np.float32)
        gain = spatial_illumination_gain(frames)
        self.assertEqual(gain.shape, (4, 96, 96))
        self.assertTrue(np.isfinite(gain).all())
        self.assertGreaterEqual(float(gain.min()), 0.80)
        self.assertLessEqual(float(gain.max()), 1.25)

    def test_uniform_illumination_is_near_identity(self):
        frames = np.full((3, 96, 96, 3), 0.35, dtype=np.float32)
        gain = spatial_illumination_gain(frames)
        self.assertLess(float(np.max(np.abs(gain - 1.0))), 0.01)

    def test_darker_region_receives_more_gain(self):
        frames = np.full((2, 96, 96, 3), 0.4, dtype=np.float32)
        frames[:, :, :48, :] *= 0.55
        gain = spatial_illumination_gain(frames)
        dark_gain = float(gain[:, 24:72, 8:32].mean())
        bright_gain = float(gain[:, 24:72, 64:88].mean())
        self.assertGreater(dark_gain, bright_gain)


if __name__ == "__main__":
    unittest.main()
