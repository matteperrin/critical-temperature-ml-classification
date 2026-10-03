"""Fast checks for explicitly separated development CV and final testing."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from src.critical_temperature import run_holdout as h
from src.critical_temperature.model_data import load_model_data


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "raw.csv"
        pd.DataFrame({"feature": np.repeat(np.arange(30), 4),
                      "critical_temp": np.tile([78., 78., 79., 76.], 30)}).to_csv(self.source, index=False)
        self.X, self.y, self.groups, self.raw = load_model_data(self.source, include_source=True)

    def recorder(self, records):
        class Estimator(BaseEstimator):
            def fit(self, X, y):
                self.fitted_ = True
                self.train = np.asarray(X).copy()
                self.labels = np.asarray(y).copy()
                records.append(self)
                return self

            def predict(self, X):
                self.valid = np.asarray(X).copy()
                return (self.valid[:, 0].astype(int) % 2).astype(int)

            def predict_proba(self, X):
                return np.tile([0.6, 0.4], (len(X), 1))
        return Estimator

    def test_split_mapping_and_fail_closed(self):
        dev, test, folds = h.create_splits(self.X, self.y, self.groups)
        again = h.create_splits(self.X, self.y, self.groups)
        np.testing.assert_array_equal(test, again[1])
        self.assertEqual(set(dev) | set(test), set(range(len(self.X))))
        self.assertTrue(set(self.groups.iloc[dev]).isdisjoint(self.groups.iloc[test]))
        seen = []
        for train, valid in folds:
            self.assertTrue(set(train) <= set(dev))
            self.assertTrue(set(valid) <= set(dev))
            self.assertTrue(set(self.groups.iloc[train]).isdisjoint(self.groups.iloc[valid]))
            seen.extend(valid)
        self.assertEqual(sorted(seen), sorted(dev))
        with self.assertRaises(ValueError):
            h.create_splits(self.X, self.y * 0, self.groups)
        with self.assertRaises(ValueError):
            h.create_splits(self.X.iloc[:4], self.y.iloc[:4], self.groups.iloc[:4])

    def test_cv_final_guards_and_shared_holdout(self):
        final_ids = []
        for name in ("elcs", "elcs_dedup", "ensemble", "ensemble_dedup", "logistic_regression", "svm", "random_forest"):
            records = []
            output = self.root / name
            with patch.object(h, "make_model", side_effect=lambda config: self.recorder(records)()):
                h.run_cv(name, output, data_path=self.source)
                self.assertEqual(len(records), 5)
                dev, test, folds = h.create_splits(self.X, self.y, self.groups)
                for record, (train, valid) in zip(records, folds):
                    np.testing.assert_array_equal(record.valid, self.X.iloc[valid].to_numpy())
                    self.assertTrue(set(record.train[:, 0]).isdisjoint(self.X.iloc[test, 0]))
                    expected = len(train) * 3 // 4 if name in ("elcs_dedup", "ensemble_dedup") else len(train)
                    self.assertEqual(len(record.train), expected)
                    if name in ("elcs_dedup", "ensemble_dedup"):
                        self.assertEqual(record.labels.sum(), expected * 2 // 3)
                self.assertFalse((output / "test_results.csv").exists())
                cv_predictions = pd.read_csv(output / "cv_predictions.csv")
                self.assertEqual(sorted(cv_predictions.row_id), sorted(dev))
                h.run_test(output)
                final = pd.read_csv(output / "test_predictions.csv")
                final_ids.append(final.row_id.tolist())
                self.assertEqual(final.row_id.tolist(), test.tolist())
                self.assertEqual(len(records), 6)
                with self.assertRaises(FileExistsError):
                    h.run_test(output)
                with self.assertRaises(FileExistsError):
                    h.run_cv(name, output, data_path=self.source)
            results = pd.read_csv(output / "cv_results.csv")
            self.assertEqual(results.fold.astype(str).tolist(), ["1", "2", "3", "4", "5", "mean", "std"])
            if name in ("logistic_regression", "svm", "random_forest"):
                self.assertIn("average_precision", results)
            else:
                self.assertNotIn("average_precision", results)
        self.assertTrue(all(ids == final_ids[0] for ids in final_ids))

    def test_changed_data_and_environment_rejected(self):
        records = []
        output = self.root / "experiment"
        with patch.object(h, "make_model", side_effect=lambda config: self.recorder(records)()):
            h.run_cv("elcs", output, data_path=self.source)
            config_path = output / "config.json"
            config = json.loads(config_path.read_text())
            config["sklearn_version"] = "changed"
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                h.run_test(output)
            config["sklearn_version"] = h.sklearn.__version__
            config_path.write_text(json.dumps(config))
            with self.source.open("a") as stream:
                stream.write("31,78\n")
            with self.assertRaises(ValueError):
                h.run_test(output)
        self.assertEqual(len(records), 5)
        with self.assertRaises(SystemExit):
            h.main(["test", "--experiment", str(output), "--iterations", "2"])

    def test_final_rejects_model_changes_after_cv(self):
        records = []
        output = self.root / "frozen"
        with patch.object(h, "make_model", side_effect=lambda config: self.recorder(records)()):
            h.run_cv("elcs", output, data_path=self.source)
            path = output / "config.json"
            config = json.loads(path.read_text())
            config["model"]["parameters"]["learning_iterations"] = 9999
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "configuration|CV artifacts"):
                h.run_test(output)
        self.assertEqual(len(records), 5)
        self.assertFalse((output / "test_started.json").exists())

    def test_final_rejects_incomplete_or_changed_cv_outputs(self):
        records = []
        output = self.root / "incomplete"
        with patch.object(h, "make_model", side_effect=lambda config: self.recorder(records)()):
            h.run_cv("elcs", output, data_path=self.source)
            (output / "cv_results.csv").write_text("")
            with self.assertRaisesRegex(ValueError, "CV artifacts"):
                h.run_test(output)
        self.assertEqual(len(records), 5)
        self.assertFalse((output / "test_started.json").exists())

    def test_missing_completion_and_interrupted_final_are_blocked(self):
        records = []
        output = self.root / "blocked"
        with patch.object(h, "make_model", side_effect=lambda config: self.recorder(records)()):
            h.run_cv("elcs", output, data_path=self.source)
            complete = output / "cv_complete.json"
            saved = complete.read_bytes()
            complete.unlink()
            with self.assertRaisesRegex(ValueError, "Complete development CV"):
                h.run_test(output)
            complete.write_bytes(saved)
            with patch.object(h, "evaluate", side_effect=RuntimeError("interrupted fit")):
                with self.assertRaisesRegex(RuntimeError, "interrupted fit"):
                    h.run_test(output)
            self.assertTrue((output / "test_started.json").exists())
            self.assertFalse((output / "test_results.csv").exists())
            with self.assertRaises(FileExistsError):
                h.run_test(output)
        self.assertEqual(len(records), 5)

    def test_data_change_rejected_with_intact_frozen_config(self):
        records = []
        output = self.root / "changed-data"
        with patch.object(h, "make_model", side_effect=lambda config: self.recorder(records)()):
            h.run_cv("elcs", output, data_path=self.source)
            with self.source.open("a") as stream:
                stream.write("31,78\n")
            with self.assertRaisesRegex(ValueError, "Dataset differs"):
                h.run_test(output)
        self.assertEqual(len(records), 5)
        self.assertFalse((output / "test_started.json").exists())

    def test_final_rejects_missing_or_changed_runtime_before_loading_data(self):
        records = []
        output = self.root / "provenance"
        with patch.object(h, "make_model", side_effect=lambda config: self.recorder(records)()):
            h.run_cv("elcs", output, data_path=self.source)
        config_path = output / "config.json"
        saved = json.loads(config_path.read_text())
        for change in ("missing", "changed"):
            config = json.loads(json.dumps(saved))
            if change == "missing":
                del config["model"]["runtime"]
                del config["model"]["library_variant"]
            else:
                config["model"]["runtime"]["source_sha256"]["DataManagement.py"] = "old"
            config_path.write_text(json.dumps(config))
            completion_path = output / "cv_complete.json"
            completion = json.loads(completion_path.read_text())
            completion["config.json"] = h.data_hash(config_path)
            completion_path.write_text(json.dumps(completion))
            with patch.object(h, "load_model_data", side_effect=AssertionError("must not access data")):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    h.run_test(output)
            self.assertFalse((output / "test_started.json").exists())

    def test_cli_and_frozen_baseline_budget(self):
        output = self.root / "baseline"
        records = []
        with patch.object(h, "make_model", side_effect=lambda config: self.recorder(records)()):
            h.main(["cv", "--model", "ensemble_dedup", "--data", str(self.source),
                    "--output-dir", str(output), "--iterations", "7",
                    "--population-size", "23", "--library-variant", "unmodified"])
        model = json.loads((output / "config.json").read_text())["model"]
        self.assertEqual(model["parameters"]["N"], 23)
        self.assertEqual(model["parameters"]["learning_iterations"], 7)
        self.assertEqual(model["library_variant"], "unmodified")
        self.assertFalse(model["runtime"]["majority_repair"]["applied"])
        self.assertEqual(len(records), 5)

    def test_conventional_configuration_does_not_read_elcs_source(self):
        with patch.object(h.elcs_runtime, "provenance", side_effect=AssertionError("not needed")):
            config = h.model_configuration("random_forest", 1, [11, 42, 73])
            self.assertNotIn("runtime", config)
            h.make_model(config)

    def test_fresh_ensemble_and_scaler_train_only(self):
        records = []
        config = h.model_configuration("ensemble", 3, [11, 42, 73])
        with patch.object(h, "elcs_model", side_effect=lambda params, **kwargs: self.recorder(records)()):
            first = h.make_model(config).fit(self.X.iloc[:8], self.y.iloc[:8])
            second = h.make_model(config).fit(self.X.iloc[8:16], self.y.iloc[8:16])
            self.assertEqual(len(records), 6)
            self.assertTrue(all(a is not b for a in first.members for b in second.members))
            np.testing.assert_array_equal(first.predict(self.X.iloc[16:20]), [0, 0, 0, 0])
        for name in ("logistic_regression", "svm"):
            records = []
            factory = "LogisticRegression" if name == "logistic_regression" else "SVC"
            with patch.object(h, factory, side_effect=lambda **kwargs: self.recorder(records)()):
                model = h.make_model(h.model_configuration(name, 3, [11, 42, 73]))
                model.fit(self.X.iloc[:8], self.y.iloc[:8])
                np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, self.X.iloc[:8].mean())
                np.testing.assert_allclose(records[0].train, StandardScaler().fit_transform(self.X.iloc[:8]))
                model.predict(self.X.iloc[8:16])
                np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, self.X.iloc[:8].mean())


if __name__ == "__main__":
    unittest.main()
