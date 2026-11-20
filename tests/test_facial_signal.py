import unittest

import numpy as np

from facial_signal import (
    extract_intensity_trace,
    extract_region_readings,
    fuse_region_signals,
    pos_signal,
)


def face_sequence(n=180, height=96, width=96, fps=30):
    y, x = np.mgrid[:height, :width]
    texture = 0.35 + 0.08 * np.sin(x / 3.0) + 0.06 * np.cos(y / 4.0)
    frames = []
    for i in range(n):
        pulse = 0.008 * np.sin(2 * np.pi * 1.2 * i / fps)
        rgb = np.stack(
            [texture + pulse, texture * 0.9 + pulse * 0.3, texture * 0.8 + pulse * 0.2],
            axis=-1,
        )
        frames.append(np.clip(rgb, 0.05, 0.85).astype(np.float32))
    return frames


class FacialSignalTests(unittest.TestCase):
    def test_regions_produce_aligned_rgb_traces_and_fused_signal(self):
        readings = extract_region_readings(face_sequence())
        self.assertEqual(set(readings), {"forehead", "left_cheek", "right_cheek"})
        self.assertTrue(all(item.rgb.shape == (180, 3) for item in readings.values()))
        fused, weights = fuse_region_signals(readings, window=48)
        self.assertEqual(fused.shape, (180,))
        self.assertTrue(np.isfinite(fused).all())
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_pos_rejects_too_short_input(self):
        with self.assertRaises(ValueError):
            pos_signal(np.ones((10, 3)), window=48)

    def test_intensity_path_accepts_monochrome_frames(self):
        frames = [np.full((32, 32), 100 + i % 2, dtype=np.uint8) for i in range(30)]
        trace = extract_intensity_trace(frames)
        self.assertEqual(trace.shape, (30,))
        self.assertTrue(np.isfinite(trace).all())

    def test_intensity_path_rejects_rgb_frames(self):
        with self.assertRaises(ValueError):
            extract_intensity_trace([np.zeros((32, 32, 3), dtype=np.uint8)] * 3)


if __name__ == "__main__":
    unittest.main()
