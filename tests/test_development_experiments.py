"""Fast checks for the bounded study's selection and execution boundaries."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.critical_temperature import run_development_experiments as study


class DevelopmentStudyTests(unittest.TestCase):
    def test_declared_sweep_selects_budget_and_runs_only_cv(self):
        calls = []

        def fake_cv(model, output, **kwargs):
            self.assertTrue((output.parent / "plan.json").is_file())
            calls.append((model, output.name, kwargs))
            output.mkdir()
            value = 0.7 if kwargs["iterations"] == 10000 else 0.6
            if model == "ensemble_dedup":
                value += 0.01
            pd.DataFrame([
                {"fold": "mean", "accuracy": value, "balanced_accuracy": value,
                 "precision": value, "recall": value, "f1": value},
                {"fold": "std", "balanced_accuracy": 0.02},
            ]).to_csv(output / "cv_results.csv", index=False)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.csv"
            data.write_text("feature,critical_temp\n1,80\n", encoding="utf-8")
            output = root / "study"
            with patch.object(study, "run_cv", side_effect=fake_cv):
                frame = study.run_study(output, data_path=data)
            self.assertEqual(len(frame), 8)
            self.assertEqual(len(calls), 8)
            self.assertEqual(calls[0][2]["library_variant"], "unmodified")
            self.assertEqual(calls[1][0], "elcs_dedup")
            self.assertEqual(calls[-1][0], "ensemble_dedup")
            self.assertEqual(calls[-1][2]["iterations"], 10000)
            self.assertEqual(calls[-1][2]["population_size"], 100)
            complete = json.loads((output / "study_complete.json").read_text())
            self.assertFalse(complete["reserved_holdout_evaluated"])
            self.assertAlmostEqual(complete["development_balanced_accuracy_difference"], 0.01)
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "completed")
            self.assertFalse(list(output.rglob("test_predictions.csv")))
            with self.assertRaises(FileExistsError):
                study.run_study(output, data_path=data)

    def test_failure_is_recorded_without_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.csv"
            data.write_text("x\n1\n", encoding="utf-8")
            output = root / "study"
            with patch.object(study, "run_cv", side_effect=RuntimeError("training failed")):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    study.run_study(output, data_path=data)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["state"], "failed_or_interrupted")
            self.assertFalse((output / "study_complete.json").exists())

    def test_source_change_blocks_next_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.csv"
            data.write_text("x\n1\n", encoding="utf-8")
            output = root / "study"
            with (patch.object(study, "source_fingerprint", side_effect=[{"x": "old"}, {"x": "new"}]),
                  patch.object(study, "run_cv") as runner):
                with self.assertRaisesRegex(RuntimeError, "Source changed"):
                    study.run_study(output, data_path=data)
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
