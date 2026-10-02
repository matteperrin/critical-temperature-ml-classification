"""Regression check for raw holdouts with training-only eLCS cleaning."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.critical_temperature import (
    run_elcs, run_logistic_regression, run_random_forest, run_svm,
)
from src.critical_temperature.model_data import load_model_data


class SharedHoldoutTests(unittest.TestCase):
    def test_runners_share_raw_holdouts_and_clean_training_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            raw = pd.DataFrame({
                "feature": np.repeat(np.arange(10), 4),
                "critical_temp": np.tile([78., 78., 79., 76.], 10),
            })
            raw.to_csv(source, index=False)
            cleaned_source = root / "cleaned.csv"
            raw.drop_duplicates().to_csv(cleaned_source, index=False)

            def loader(path=None, **kwargs):
                selected = source if path is None else cleaned_source
                return load_model_data(selected, **kwargs)

            runs = []
            scaled_folds = []

            class Estimator:
                def __init__(self, **parameters):
                    models.append(self)

                def fit(self, values, labels):
                    self.train = np.asarray(values).copy()
                    self.labels = np.asarray(labels).copy()
                    return self

                def predict(self, values):
                    self.test = np.asarray(values).copy()
                    return np.zeros(len(values), dtype=int)

                def predict_proba(self, values):
                    return np.tile([0.7, 0.3], (len(values), 1))

            class Scaler:
                def __init__(self):
                    scaled_folds.append(self)

                def fit_transform(self, values):
                    self.train = np.asarray(values).copy()
                    return np.asarray(values)

                def transform(self, values):
                    self.test = np.asarray(values).copy()
                    return np.asarray(values)

            variants = [
                (run_elcs, "eLCS", []),
                (run_elcs, "eLCS", ["--data", "preprocessed"]),
                (run_random_forest, "RandomForestClassifier", None),
                (run_logistic_regression, "LogisticRegression", None),
                (run_svm, "SVC", None),
            ]
            for module, estimator, argv in variants:
                models = []
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(module, "load_model_data", loader))
                    stack.enter_context(patch.object(module, estimator, Estimator))
                    if module is run_elcs:
                        stack.enter_context(patch.object(module, "REPORT_DIR", root))
                    else:
                        stack.enter_context(patch.object(module, "REPORT_PATH", root / "result.csv"))
                    if hasattr(module, "StandardScaler"):
                        stack.enter_context(patch.object(module, "StandardScaler", Scaler))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if argv is None:
                        module.main()
                    else:
                        module.main(argv)
                self.assertEqual(len(models), 5)
                runs.append(models)

            baseline, cleaned = runs[:2]
            for models in runs[1:]:
                for original, actual in zip(baseline, models):
                    np.testing.assert_array_equal(actual.test, original.test)
                    self.assertEqual(actual.test.shape[1], 1)
                    self.assertTrue(set(actual.train[:, 0]).isdisjoint(actual.test[:, 0]))
            for original, clean in zip(baseline, cleaned):
                self.assertEqual(len(clean.train), len(original.train) * 3 // 4)
                # Both 78 and 79 K survive; only the repeated 78 K row is removed.
                self.assertEqual(int((clean.labels == 1).sum()), len(clean.train) * 2 // 3)
            for models in runs[2:]:
                for original, actual in zip(baseline, models):
                    np.testing.assert_array_equal(actual.train, original.train)
            self.assertEqual(len(scaled_folds), 10)
            for scaler in scaled_folds:
                self.assertTrue(set(scaler.train[:, 0]).isdisjoint(scaler.test[:, 0]))
            self.assertEqual(sum(len(model.test) for model in baseline), len(raw))
            pd.testing.assert_frame_equal(pd.read_csv(source), raw)


if __name__ == "__main__":
    unittest.main()
