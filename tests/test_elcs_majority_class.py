"""Regression checks for the bundled eLCS majority-class fallback."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parents[1] / "third_party/scikit-eLCS/skeLCS"


def load_bundled_module(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DataManagement = load_bundled_module("DataManagement").DataManagement
Prediction = load_bundled_module("Prediction").Prediction


class MajorityClassTests(unittest.TestCase):
    def make_data(self, labels):
        features = np.arange(len(labels), dtype=float).reshape(-1, 1)
        return DataManagement(
            features, np.asarray(labels), SimpleNamespace(discrete_attribute_limit=10)
        )

    def test_majority_follows_count_not_label_order(self):
        for labels, expected in [
            ([0, 0, 0, 1], 0),
            ([1, 0, 0, 0], 0),
            ([0, 1, 1, 1], 1),
            ([2, 2, 2, 9], 2),
        ]:
            with self.subTest(labels=labels):
                self.assertEqual(self.make_data(labels).majorityClass, expected)

    def test_tied_counts_choose_first_observed_label(self):
        for labels in ([0, 1], [1, 0]):
            with self.subTest(labels=labels):
                self.assertEqual(self.make_data(labels).majorityClass, labels[0])

    def test_no_matching_rule_predicts_training_majority(self):
        data = self.make_data([0, 0, 0, 1])
        prediction = Prediction(
            SimpleNamespace(env=SimpleNamespace(formatData=data)),
            SimpleNamespace(matchSet=[], popSet=[]),
        )
        self.assertFalse(prediction.hasMatch)
        self.assertEqual(prediction.getDecision(), 0)


if __name__ == "__main__":
    unittest.main()
