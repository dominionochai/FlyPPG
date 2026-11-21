import unittest

import numpy as np

import ss_preprocess


class SurroundGainTests(unittest.TestCase):
    def make_video(self, background_gain, foreground_gain=None):
        frames = len(background_gain)
        video = np.empty((frames, 96, 96, 3), dtype=np.float32)
        video[:] = 0.20
        video[:, 24:72, 24:72, :] = 0.50
        video[:, :, :, :] *= np.asarray(background_gain, dtype=np.float32)[:, None, None, None]
        if foreground_gain is not None:
            # Replace the foreground with an independently varying signal.
            video[:, 24:72, 24:72, :] = (
                0.50 * np.asarray(foreground_gain, dtype=np.float32)[:, None, None, None]
            )
        return video

    def test_static_video_uses_identity_gain(self):
        gain = np.ones(300, dtype=np.float32)
        result = ss_preprocess.surround_gain(self.make_video(gain))
        np.testing.assert_allclose(result, 1.0)

    def test_shared_illumination_is_estimated_from_background(self):
        t = np.arange(300, dtype=np.float32) / 30.0
        shared = 1.0 + 0.12 * np.sin(2 * np.pi * 0.12 * t)
        result = ss_preprocess.surround_gain(self.make_video(shared))
        # Correction should reduce shared flicker without rescaling the whole clip.
        self.assertLess(np.std(result), np.std(shared / np.median(shared)))
        self.assertLessEqual(float(np.max(np.abs(result - 1.0))), 0.051)
        self.assertGreater(float(np.std(result)), 0.001)

    def test_face_only_flicker_does_not_trigger_background_correction(self):
        t = np.arange(300, dtype=np.float32) / 30.0
        background = np.ones_like(t)
        foreground = 1.0 + 0.15 * np.sin(2 * np.pi * 0.17 * t)
        result = ss_preprocess.surround_gain(self.make_video(background, foreground))
        np.testing.assert_allclose(result, 1.0)

    def test_uncorrelated_background_change_does_not_trigger_correction(self):
        t = np.arange(300, dtype=np.float32) / 30.0
        background = 1.0 + 0.12 * np.sin(2 * np.pi * 0.11 * t)
        foreground = 1.0 + 0.12 * np.sin(2 * np.pi * 0.29 * t + 0.7)
        result = ss_preprocess.surround_gain(self.make_video(background, foreground))
        np.testing.assert_allclose(result, 1.0)


if __name__ == "__main__":
    unittest.main()
