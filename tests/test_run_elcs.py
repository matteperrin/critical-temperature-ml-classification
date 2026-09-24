"""Fast checks for the baseline runner using a stand-in estimator."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.critical_temperature import run_elcs


class BaselineRunnerTests(unittest.TestCase):
    def test_folds_models_and_report(self):
        # Each group has repeated features and both target classes.
        groups = pd.Series(np.repeat(np.arange(10), 2))
        X = pd.DataFrame({"feature": groups})
        y = pd.Series(np.tile([0, 1], 10))
        models = []

        class Estimator:
            def __init__(self, **parameters):
                self.parameters = parameters
                self.train = None
                self.test = None
                models.append(self)

            def fit(self, values, labels):
                self.train = values[:, 0].copy()
                self.labels = labels.copy()

            def predict(self, values):
                self.test = values[:, 0].copy()
                return (self.test % 2).astype(int)

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "reports" / "baseline.csv"
            with (
                patch.object(run_elcs, "load_model_data", return_value=(X, y, groups)),
                patch.object(run_elcs, "eLCS", Estimator),
                patch.object(run_elcs, "REPORT_PATH", report),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_elcs.main()
            output = pd.read_csv(report)

        self.assertEqual(len(models), 5)
        self.assertEqual(output["fold"].tolist(), ["1", "2", "3", "4", "5", "mean", "std"])
        self.assertEqual(output["n_splits"].tolist(), [5] * 7)
        for model in models:
            self.assertTrue(set(model.train).isdisjoint(model.test))
            self.assertEqual(len(model.train) + len(model.test), len(X))
            self.assertEqual(len(model.labels), len(model.train))
            self.assertEqual(model.parameters, {
                "learning_iterations": run_elcs.LEARNING_ITERATIONS,
                "N": run_elcs.POPULATION_SIZE,
                "random_state": run_elcs.RANDOM_STATE,
            })
        np.testing.assert_array_equal(
            np.sort(np.concatenate([model.test for model in models])),
            np.sort(X["feature"].to_numpy()),
        )
        folds = output.iloc[:5]
        self.assertEqual(folds["training_rows"].tolist(), [len(m.train) for m in models])
        self.assertEqual(folds["test_rows"].tolist(), [len(m.test) for m in models])
        for metric in run_elcs.METRICS:
            for index, statistic in [(5, "mean"), (6, "std")]:
                self.assertAlmostEqual(
                    output.loc[index, metric], folds[metric].agg(statistic), places=5
                )
        # Both classes occur equally often within every test group.
        np.testing.assert_allclose(folds["accuracy"], 0.5)
        np.testing.assert_allclose(folds["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
