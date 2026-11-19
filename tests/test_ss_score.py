import json
import os
import tempfile
import unittest

import ss_score


class ReadingParserTests(unittest.TestCase):
    def test_protobuf_camel_case_and_metadata(self):
        payload = {
            "cardio": {
                "pulseRate": [
                    {"value": 72.5, "confidence": 0.91, "stable": True, "timestamp": "12000000"},
                    {"value": 73.0, "confidence": 0.4, "stable": False, "timestamp": "14000000"},
                ]
            }
        }
        line = "Got edge metrics at 12000000 microseconds: " + json.dumps(payload)
        parsed = ss_score.readings(line)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["value"], 72.5)
        self.assertEqual(parsed[0]["confidence"], 0.91)
        self.assertIs(parsed[0]["stable"], True)

    def test_snake_case_and_callback_timestamp_fallback(self):
        payload = {"cardio": {"pulse_rate": [{"value": 68.0}]}}
        line = "Got edge metrics at 9000000 microseconds: " + json.dumps(payload)
        parsed = ss_score.readings(line)
        self.assertEqual(parsed[0]["timestamp_us"], 9000000)

    def test_malformed_line_is_ignored(self):
        parsed = ss_score.readings("Got edge metrics at 900 microseconds: {broken}")
        self.assertEqual(parsed, [])

    def test_confidence_and_stability_filters(self):
        samples = [
            {"value": 70.0, "confidence": 0.2, "stable": False},
            {"value": 72.0, "confidence": 0.9, "stable": True},
        ]
        self.assertEqual(ss_score.estimate(samples, 0.5, True), 72.0)

    def test_pairwise_comparison_does_not_require_other_arms(self):
        arm = {"a": 1.0, "b": 2.0}
        control = {"a": 2.0, "b": 3.0, "c": 4.0}
        result = ss_score.bootstrap_delta(arm, control, __import__("numpy").random.default_rng(0), 100)
        self.assertEqual(result["n_paired_videos"], 2)
        self.assertEqual(result["delta_mae_bpm"], -1.0)


if __name__ == "__main__":
    unittest.main()
