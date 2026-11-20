import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import benchmark_status as bs  # noqa: E402
import benchmark_gate  # noqa: E402

ROWS = [
    {"reference_bpm": 70.0, "smartspectra_bpm": None},
    {"reference_bpm": 72.0, "smartspectra_bpm": None},
]


class SampleFilteringTests(unittest.TestCase):
    def test_zero_confidence_and_nonpositive_are_not_readings(self):
        raw = [(1.0, 0.0, 0.0), (2.0, 75.0, 0.0), (3.0, -1, 0.9), (13.0, 71.0, 0.8), (14.0, 72.0, None)]
        usable, dropped = bs.split_samples(raw)
        self.assertEqual(usable, [(13.0, 71.0), (14.0, 72.0)])
        self.assertEqual(dropped, 3)

    def test_long_dict_timestamps_and_values_are_parsed(self):
        self.assertEqual(bs._num({"low": 12000000, "high": 0, "unsigned": False}), 12000000.0)
        self.assertEqual(bs._num({"low": 0, "high": 1, "unsigned": True}), 4294967296.0)
        self.assertIsNone(bs._num(None))
        self.assertIsNone(bs._num(float("nan")))


class MissingBaselineTests(unittest.TestCase):
    def test_no_samples_is_invalid_not_zero(self):
        arm = bs.smartspectra_arm(ROWS, 0, 0)
        self.assertEqual(arm["status"], bs.INVALID_NO_OUTPUT)
        self.assertIsNone(arm["mae_bpm"])
        self.assertIsNone(arm["rmse_bpm"])
        self.assertEqual(arm["windows_scored"], 0)
        self.assertEqual(arm["coverage_pct"], 0.0)

    def test_only_discarded_samples_is_still_invalid(self):
        usable, dropped = bs.split_samples([(1.0, 0.0, 0.0), (5.0, 80.0, 0.0)])
        arm = bs.smartspectra_arm(ROWS, len(usable), dropped)
        self.assertEqual(arm["status"], bs.INVALID_NO_OUTPUT)
        self.assertEqual(arm["discarded_zero_confidence_or_nonpositive"], 2)

    def test_samples_outside_all_windows_are_invalid(self):
        arm = bs.smartspectra_arm(ROWS, 3, 0)  # usable samples exist but no row has smartspectra_bpm
        self.assertEqual(arm["status"], bs.INVALID_NO_ALIGNED_WINDOWS)
        self.assertIsNone(arm["mae_bpm"])

    def test_valid_arm_scores(self):
        rows = [{"reference_bpm": 70.0, "smartspectra_bpm": 72.0}, {"reference_bpm": 72.0, "smartspectra_bpm": 69.0}]
        arm = bs.smartspectra_arm(rows, 4, 1)
        self.assertEqual(arm["status"], bs.VALID)
        self.assertAlmostEqual(arm["mae_bpm"], 2.5)
        self.assertEqual(arm["within_5_bpm_pct_of_all_windows"], 100.0)


class GateTests(unittest.TestCase):
    def _report(self, status):
        return {"arms": {"raw_POS": {"windows_scored": 5, "mae_bpm": 1.0}, "FlyPPG_v3": {"windows_scored": 5, "mae_bpm": 1.0},
                         "SmartSpectra": {"status": status}}}

    def test_exit_codes(self):
        self.assertEqual(bs.gate_exit_code(self._report(bs.VALID)), 0)
        self.assertEqual(bs.gate_exit_code(self._report(bs.INVALID_NO_OUTPUT)), 1)
        self.assertEqual(bs.gate_exit_code({"arms": {}}), 1)

    def test_gate_cli_fails_on_invalid_and_missing_report(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.json")
            json.dump(self._report(bs.INVALID_NO_OUTPUT), open(p, "w"))
            self.assertEqual(benchmark_gate.main(p), 1)
            self.assertEqual(benchmark_gate.main(os.path.join(d, "missing.json")), 1)
            json.dump(self._report(bs.VALID), open(p, "w"))
            self.assertEqual(benchmark_gate.main(p), 0)


if __name__ == "__main__":
    unittest.main()
