"""Tests for the Phase II model-data loader."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.critical_temperature.model_data import load_model_data


class ModelDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "train.csv"
        self.data = pd.DataFrame(
            {
                "feature": [1.0, 1.0, 2.0],
                "critical_temp": [76.0, 78.0, 77.0],
                # These labels are deliberately wrong; the loader should ignore them.
                "above_77k": [1, 0, 1],
                "material": ["A", "B", "C"],
            }
        )

    def test_target_predictors_groups_and_read_only_loading(self):
        self.data.to_csv(self.path, index=False)
        original = self.path.read_bytes()
        X, y, groups = load_model_data(self.path)
        self.assertEqual(X.columns.tolist(), ["feature"])
        self.assertEqual(y.tolist(), [0, 1, 0])
        self.assertEqual(len(X), len(self.data))
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])
        self.assertTrue(X.index.equals(y.index))
        self.assertTrue(X.index.equals(groups.index))
        self.assertEqual(self.path.read_bytes(), original)

    def test_raw_data_needs_no_saved_label_or_material(self):
        self.data.drop(columns=["above_77k", "material"]).to_csv(self.path, index=False)
        X, y, _ = load_model_data(self.path)
        self.assertEqual(X.shape, (3, 1))
        self.assertEqual(y.tolist(), [0, 1, 0])

    def test_invalid_data_is_rejected(self):
        cases = [
            self.data.iloc[:0],
            self.data.drop(columns="critical_temp"),
            self.data.drop(columns="feature"),
        ]
        for column, value in [
            ("feature", "invalid"),
            ("feature", float("inf")),
            ("feature", float("nan")),
            ("critical_temp", float("nan")),
            ("critical_temp", -float("inf")),
            ("critical_temp", 0),
        ]:
            cases.append(self.data.assign(**{column: value}))
        for number, data in enumerate(cases):
            with self.subTest(case=number):
                data.to_csv(self.path, index=False)
                with self.assertRaises(ValueError):
                    load_model_data(self.path)


if __name__ == "__main__":
    unittest.main()
