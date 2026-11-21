import json
import os
import tempfile
import unittest

from ss_window_score import load_reference, readings, score_window, evaluate


class TimestampAlignedScoringTests(unittest.TestCase):
    def test_extracts_timestamps_and_metadata(self):
        payload = {"cardio": {"pulseRate": [
            {"value": 72, "timestamp": "12000000", "confidence": 0.9, "stable": True},
            {"value": 75, "timestamp": "18000000", "confidence": 0.8, "stable": False},
        ]}}
        line = "Got edge metrics at 12000000 microseconds: " + json.dumps(payload)
        samples = readings(line)
        self.assertEqual([x["timestamp_us"] for x in samples], [12000000, 18000000])
        self.assertEqual(samples[0]["confidence"], 0.9)

    def test_window_selects_only_samples_inside_interval(self):
        samples = [
            {"value": 60.0, "timestamp_us": 4_000_000, "confidence": 0.9, "stable": True},
            {"value": 72.0, "timestamp_us": 6_000_000, "confidence": 0.9, "stable": True},
            {"value": 74.0, "timestamp_us": 8_000_000, "confidence": 0.9, "stable": True},
            {"value": 90.0, "timestamp_us": 10_000_000, "confidence": 0.9, "stable": True},
        ]
        self.assertEqual(score_window(samples, 5, 10), 73.0)

    def test_confidence_and_stability_filters_preserve_missing(self):
        samples = [
            {"value": 72.0, "timestamp_us": 6_000_000, "confidence": 0.4, "stable": False},
            {"value": 74.0, "timestamp_us": 8_000_000, "confidence": 0.9, "stable": True},
        ]
        self.assertEqual(score_window(samples, 5, 10, min_confidence=0.8, require_stable=True), 74.0)
        self.assertIsNone(score_window(samples, 5, 7, min_confidence=0.8, require_stable=True))

    def test_reference_rejects_overlapping_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ref.csv")
            with open(path, "w") as handle:
                handle.write("video,start_s,end_s,hr_bpm\na,0,10,72\na,9,19,74\n")
            with self.assertRaises(ValueError):
                load_reference(path)

    def test_evaluation_reports_coverage_and_paired_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ("control", "candidate"):
                os.makedirs(os.path.join(tmp, arm))
            payload = {"cardio": {"pulseRate": [
                {"value": 72.0, "timestamp": "2000000"},
                {"value": 72.0, "timestamp": "7000000"},
            ]}}
            line = "Got edge metrics at 2000000 microseconds: " + json.dumps(payload)
            for arm in ("control", "candidate"):
                with open(os.path.join(tmp, arm, "clip.txt"), "w") as handle:
                    handle.write(line)
            ref = {"clip": [{"start_s": 0, "end_s": 10, "hr_bpm": 72}]}
            result = evaluate(tmp, ref, bootstrap=100)
            self.assertEqual(result["arms"]["control"]["coverage_pct"], 100.0)
            self.assertEqual(result["arms"]["candidate"]["mae_bpm"], 0.0)
            self.assertEqual(result["paired_vs_control"]["candidate"]["paired_videos"], 1)


if __name__ == "__main__":
    unittest.main()
