"""Tests for the Random Forest experiment."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.critical_temperature import run_random_forest


class RandomForestTests(unittest.TestCase):

    def test_five_fold_experiment(self):

        # Create a small artificial dataset.
        groups = pd.Series(np.repeat(np.arange(10), 2))
        X = pd.DataFrame({"feature": groups})
        y = pd.Series(np.tile([0, 1], 10))

        models = []

        # Substitute model to test the experiment quickly.
        class FakeForest:

            def __init__(self, **parameters):
                self.parameters = parameters
                models.append(self)

            def fit(self, X_train, y_train):
                self.train_groups = set(X_train["feature"])
                return self

            def predict(self, X_test):
                self.test_groups = set(X_test["feature"])
                return np.zeros(len(X_test), dtype=int)

            def predict_proba(self, X_test):
                return np.tile([0.7, 0.3], (len(X_test), 1))

        with tempfile.TemporaryDirectory() as directory:

            report = Path(directory) / "results.csv"

            with (
                patch.object(
                    run_random_forest,
                    "load_model_data",
                    return_value=(X, y, groups),
                ),
                patch.object(
                    run_random_forest,
                    "RandomForestClassifier",
                    FakeForest,
                ),
                patch.object(
                    run_random_forest,
                    "REPORT_PATH",
                    report,
                ),
            ):
                run_random_forest.main()

            results = pd.read_csv(report)

        # Verify that five separate models were trained.
        self.assertEqual(len(models), 5)

        # Verify that training and testing groups never overlap.
        for model in models:
            self.assertTrue(
                model.train_groups.isdisjoint(model.test_groups)
            )

        # Every group should appear in exactly one test fold.
        all_test_groups = [
            group
            for model in models
            for group in model.test_groups
        ]

        self.assertEqual(
            sorted(all_test_groups),
            list(range(10)),
        )

        # Verify that results contain five folds and two summaries.
        self.assertEqual(len(results), 7)

        self.assertEqual(
            results["fold"].astype(str).tolist(),
            ["1", "2", "3", "4", "5", "mean", "std"],
        )

        # Verify that the expected metrics were recorded.
        for metric in run_random_forest.METRICS:
            self.assertIn(metric, results.columns)


if __name__ == "__main__":
    unittest.main()
    