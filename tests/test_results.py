"""Ensure seven-threshold aggregation and complete-fold checks are explicit."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_results import THRESHOLDS, metrics, parse_log


class ResultTest(unittest.TestCase):
    def test_selects_seven_thresholds(self):
        line = "Avg. mAP 99.0 (%) mAP@0.35 99.0 (%) " + " ".join(
            f"mAP@{t:.1f} {10 * i:.2f} (%)" for i, t in enumerate(THRESHOLDS)
        )
        result = metrics(line)
        self.assertEqual(sum(result[t] for t in THRESHOLDS) / 7, 30.)
        self.assertIsNone(metrics("mAP@0.3 90.0 (%)"))

    def test_preserves_fold_results_and_detects_duplicates(self):
        line = " ".join(f"mAP@{t} 80.0 (%)" for t in THRESHOLDS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text(f"FINAL RESULTS SUBJECT 0\n{line}\nFINAL AVERAGED RESULTS:\n{line}\n")
            folds, final, duplicates = parse_log(path)
            self.assertEqual(set(folds), {0})
            self.assertIsNotNone(final)
            self.assertFalse(duplicates)
            with path.open("a") as handle:
                handle.write(f"FINAL RESULTS SUBJECT 0\n{line}\n")
            self.assertTrue(parse_log(path)[2])


if __name__ == "__main__":
    unittest.main()
