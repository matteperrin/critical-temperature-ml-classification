"""Comparison outputs require compatible, complete evaluation records."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.critical_temperature import compare_models as c
from src.critical_temperature import run_holdout as h


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "raw.csv"
        self.raw = pd.DataFrame({
            "feature": np.repeat(np.arange(30), 4),
            "critical_temp": np.tile([78., 78., 79., 76.], 30),
        })
        self.raw.to_csv(self.source, index=False)

    def experiment(self, name, *, final=False, source=None, directory=None, variant="corrected"):
        output = self.root / (directory or name)

        class Model:
            def __init__(self):
                self.members = [self, self, self]

            def export_final_rule_population(self, **kwargs):
                pd.DataFrame({"Accuracy": [0.5], "Match Count": [1], "Fitness": [0.1]}).to_csv(
                    kwargs["filename"], index=False)

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):
                return np.tile([0.6, 0.4], (len(X), 1))

        with (patch.object(h, "make_model", return_value=Model()),
              contextlib.redirect_stdout(io.StringIO())):
            h.run_cv(name, output, data_path=source or self.source, library_variant=variant)
            if final:
                h.run_test(output)
        return output

    def test_final_requires_completion_and_rejects_consistent_rewritten_outputs(self):
        experiment = self.experiment("elcs_dedup", final=True)
        manifest = experiment / "test_complete.json"
        original = manifest.read_bytes()
        manifest.unlink()
        with self.assertRaisesRegex(ValueError, "completion|Complete final"):
            c.load_experiment(experiment, "test")
        manifest.write_bytes(original)
        predictions = pd.read_csv(experiment / "test_predictions.csv")
        predictions["y_pred"] = 1
        predictions.to_csv(experiment / "test_predictions.csv", index=False)
        results = pd.read_csv(experiment / "test_results.csv")
        for name, value in c.metrics(predictions).items():
            results.loc[0, name] = value
        tn, fp, fn, tp = c.confusion_matrix(predictions.y_true, predictions.y_pred, labels=[0, 1]).ravel()
        results.loc[0, ["true_negatives", "false_positives", "false_negatives", "true_positives"]] = [tn, fp, fn, tp]
        results.to_csv(experiment / "test_results.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Final artifact"):
            c.load_experiment(experiment, "test")

    def test_final_rejects_missing_rule_exports_even_with_rewritten_manifest(self):
        experiment = self.experiment("ensemble_dedup", final=True)
        path = experiment / "test_complete.json"
        manifest = json.loads(path.read_text())
        del manifest["rules/metadata.json"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "rule exports|Rules"):
            c.load_experiment(experiment, "test")

    def test_primary_rejects_other_budgets_seeds_and_statistical_settings(self):
        baseline = self.experiment("elcs_dedup", final=True)
        improved = self.experiment("ensemble_dedup", final=True)
        config, predictions = c.load_experiment(improved, "test")
        for field, value in (("learning_iterations", 10000), ("N", 1000), ("seeds", [1, 2, 3])):
            changed = json.loads(json.dumps(config))
            changed["model"]["parameters"][field] = value
            actual_loader = c.load_experiment
            def loader(directory, stage):
                return (changed, predictions) if directory == improved else actual_loader(directory, stage)
            with patch.object(c, "load_experiment", side_effect=loader):
                with self.assertRaisesRegex(ValueError, "selected settings"):
                    c.statistical_test(baseline, improved, output=self.root / "wrong_settings.json")
        for settings in ({"permutations": 999}, {"random_state": 7}):
            with self.assertRaisesRegex(ValueError, "statistical recipe"):
                c.statistical_test(baseline, improved, output=self.root / "wrong_recipe.json", **settings)

    def test_probability_scores_round_trip_without_changing_rank_metrics(self):
        experiment = self.experiment("random_forest")
        predictions = pd.read_csv(experiment / "cv_predictions.csv")
        predictions["score"] = np.where(predictions.y_true.eq(1), np.nextafter(0.1, 1), 0.1)
        predictions.to_csv(experiment / "cv_predictions.csv", index=False)
        results = pd.read_csv(experiment / "cv_results.csv", dtype={"fold": str}).set_index("fold")
        fold_metrics = []
        for fold, rows in predictions.groupby("fold"):
            values = c.metrics(rows)
            fold_metrics.append(values)
            for name, value in values.items():
                results.loc[str(fold), name] = value
        values = pd.DataFrame(fold_metrics)
        for name in values:
            results.loc["mean", name] = values[name].mean()
            results.loc["std", name] = values[name].std(ddof=1)
        results.reset_index().to_csv(experiment / "cv_results.csv", index=False)
        self.rehash(experiment, "cv_predictions.csv")
        self.rehash(experiment, "cv_results.csv")
        _, loaded = c.load_experiment(experiment, "cv")
        self.assertEqual(loaded.score.nunique(), 2)
        self.assertEqual(c.metrics(loaded)["roc_auc"], 1)

    def rehash(self, experiment, artifact):
        path = experiment / "cv_complete.json"
        completion = json.loads(path.read_text())
        completion[artifact] = h.data_hash(experiment / artifact)
        path.write_text(json.dumps(completion))

    def test_variants_have_explicit_readable_names_and_no_elcs_score_metrics(self):
        raw = self.experiment("elcs", variant="unmodified")
        dedup = self.experiment("elcs_dedup")
        table = c.comparison_table([raw, dedup], stage="cv", output=self.root / "variants.csv")
        self.assertEqual(table.source_variant.tolist(), ["unmodified", "corrected"])
        self.assertIn("Unmodified", table.display_name.iloc[0])
        self.assertIn("Corrected", table.display_name.iloc[1])
        self.assertTrue(table.roc_auc.isna().all())
        self.assertIn("balanced_accuracy", c.markdown_summary(table))

    def test_rejects_changed_runtime_even_with_rehashed_configuration(self):
        experiment = self.experiment("elcs")
        path = experiment / "config.json"
        config = json.loads(path.read_text())
        config["model"]["runtime"]["package_version"] = "tampered"
        path.write_text(json.dumps(config))
        self.rehash(experiment, "config.json")
        with self.assertRaisesRegex(ValueError, "runtime|provenance"):
            c.load_experiment(experiment, "cv")

    def test_rejects_rehashed_wrong_labels_folds_and_cv_results(self):
        experiment = self.experiment("elcs")
        path = experiment / "cv_predictions.csv"
        original = pd.read_csv(path)
        for column, value in (("y_true", 1 - original.y_true.iloc[0]), ("fold", 99)):
            rows = original.copy()
            rows.loc[0, column] = value
            rows.to_csv(path, index=False)
            self.rehash(experiment, "cv_predictions.csv")
            with self.assertRaisesRegex(ValueError, "labels|folds"):
                c.load_experiment(experiment, "cv")
        original.to_csv(path, index=False)
        self.rehash(experiment, "cv_predictions.csv")
        result_path = experiment / "cv_results.csv"
        results = pd.read_csv(result_path)
        results.loc[0, "accuracy"] = 0.99
        results.to_csv(result_path, index=False)
        self.rehash(experiment, "cv_results.csv")
        with self.assertRaisesRegex(ValueError, "results"):
            c.load_experiment(experiment, "cv")

    def test_primary_comparison_rejects_unmodified_variant(self):
        baseline = self.experiment("elcs_dedup", final=True, variant="unmodified")
        improved = self.experiment("ensemble_dedup", final=True)
        with self.assertRaisesRegex(ValueError, "corrected"):
            c.statistical_test(baseline, improved, output=self.root / "wrong_variant.json")

    def test_rejects_missing_completion_changed_source_and_saved_splits(self):
        experiment = self.experiment("elcs")
        completion = experiment / "cv_complete.json"
        saved = completion.read_bytes()
        completion.unlink()
        with self.assertRaisesRegex(ValueError, "Complete"):
            c.load_experiment(experiment, "cv")
        completion.write_bytes(saved)
        source = self.source.read_bytes()
        self.source.write_bytes(source + b"\n")
        with self.assertRaisesRegex(ValueError, "dataset"):
            c.load_experiment(experiment, "cv")
        self.source.write_bytes(source)
        path = experiment / "splits.csv"
        rows = pd.read_csv(path)
        rows.loc[0, "cv_fold"] = 99
        rows.to_csv(path, index=False)
        self.rehash(experiment, "splits.csv")
        with self.assertRaisesRegex(ValueError, "assignments"):
            c.load_experiment(experiment, "cv")

    def test_stats_cli_validates_final_only_and_never_trains(self):
        baseline = self.experiment("elcs_dedup", final=True)
        improved = self.experiment("ensemble_dedup", final=True)
        output = self.root / "stats_cli.json"
        with patch.object(h, "make_model", side_effect=AssertionError("No training")):
            c.main(["stats", "--baseline", str(baseline), "--improved", str(improved),
                    "--output", str(output)])
        result = json.loads(output.read_text())
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["alpha"], 0.05)
        self.assertEqual(result["draws"], 1)

    def test_table_cli_never_trains_and_does_not_overwrite(self):
        experiment = self.experiment("elcs")
        output = self.root / "cli.csv"
        args = ["table", "--stage", "cv", "--experiments", str(experiment), "--output", str(output)]
        with patch.object(h, "make_model", side_effect=AssertionError("No training")):
            c.main(args)
        self.assertTrue(output.exists())
        with self.assertRaises(FileExistsError):
            c.main(args)

    def test_cv_table_covers_six_models_without_final_evaluation(self):
        models = ("elcs", "elcs_dedup", "ensemble_dedup", "logistic_regression", "svm", "random_forest")
        experiments = [self.experiment(name) for name in models]
        output = self.root / "cv_table.csv"
        table = c.comparison_table(experiments, stage="cv", output=output)
        self.assertEqual(table.model.tolist(), list(models))
        np.testing.assert_allclose(table.balanced_accuracy, 0.5)
        np.testing.assert_allclose(table.balanced_accuracy_std, 0)
        self.assertTrue(table.evaluation_rows.eq(96).all())
        self.assertTrue(table.aggregation.eq("unweighted_cv_mean").all())
        self.assertTrue(table.loc[:2, "average_precision"].isna().all())
        self.assertTrue(table.loc[3:, "average_precision"].notna().all())
        for directory in experiments:
            self.assertFalse((directory / "test_started.json").exists())
        with self.assertRaises(FileExistsError):
            c.comparison_table(experiments, stage="cv", output=output)

    def test_final_table_and_paired_test_use_same_rows_and_groups(self):
        baseline = self.experiment("elcs_dedup", final=True)
        improved = self.experiment("ensemble_dedup", final=True)
        table = c.comparison_table([baseline, improved], stage="test", output=self.root / "test_table.csv")
        self.assertTrue(table.evaluation_rows.eq(24).all())
        self.assertTrue(table.aggregation.eq("heldout").all())
        self.assertEqual(table.true_negatives.tolist(), [6, 6])
        output = self.root / "statistical_test.json"
        result = c.statistical_test(baseline, improved, output=output)
        self.assertEqual(result["p_value"], 1)
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["group_count"], 6)
        self.assertEqual(json.loads(output.read_text())["p_value"], 1)
        with self.assertRaises(FileExistsError):
            c.statistical_test(baseline, improved, output=output)

    def test_rejects_missing_final_results_and_incompatible_experiments(self):
        baseline = self.experiment("elcs_dedup")
        improved = self.experiment("ensemble_dedup")
        with self.assertRaisesRegex(ValueError, "final|Final"):
            c.statistical_test(baseline, improved, output=self.root / "not_final.json")
        changed_source = self.root / "changed.csv"
        self.raw.assign(feature=self.raw.feature + 100).to_csv(changed_source, index=False)
        changed = self.experiment("elcs", source=changed_source, directory="changed")
        with self.assertRaisesRegex(ValueError, "dataset|compatible"):
            c.comparison_table([baseline, changed], stage="cv", output=self.root / "mixed.csv")
        self.assertFalse((self.root / "mixed.csv").exists())

    def test_rejects_empty_final_results_and_prediction_disagreement(self):
        experiment = self.experiment("elcs_dedup", final=True)
        result_path = experiment / "test_results.csv"
        saved = result_path.read_bytes()
        result_path.write_text("")
        with self.assertRaisesRegex(ValueError, "final results|Final results"):
            c.comparison_table([experiment], stage="test", output=self.root / "empty.csv")
        result_path.write_bytes(saved)
        prediction_path = experiment / "test_predictions.csv"
        predictions = pd.read_csv(prediction_path)
        predictions.loc[0, "y_pred"] = 1 - predictions.loc[0, "y_pred"]
        predictions.to_csv(prediction_path, index=False)
        with self.assertRaisesRegex(ValueError, "final results|Final results"):
            c.comparison_table([experiment], stage="test", output=self.root / "disagrees.csv")

    def test_rejects_inconsistent_recorded_split_even_for_one_experiment(self):
        experiment = self.experiment("elcs")
        path = experiment / "config.json"
        config = json.loads(path.read_text())
        config["split"]["random_state"] = 43
        path.write_text(json.dumps(config))
        completion_path = experiment / "cv_complete.json"
        completion = json.loads(completion_path.read_text())
        completion["config.json"] = h.data_hash(path)
        completion_path.write_text(json.dumps(completion))
        with self.assertRaisesRegex(ValueError, "split|Split"):
            c.comparison_table([experiment], stage="cv", output=self.root / "split.csv")

    def test_rejects_changed_cv_artifacts_and_duplicate_final_ids(self):
        experiment = self.experiment("elcs", final=True)
        final_path = experiment / "test_predictions.csv"
        predictions = pd.read_csv(final_path)
        predictions.loc[1, "row_id"] = predictions.loc[0, "row_id"]
        predictions.to_csv(final_path, index=False)
        with self.assertRaisesRegex(ValueError, "row|duplicate"):
            c.comparison_table([experiment], stage="test", output=self.root / "duplicate.csv")
        (experiment / "cv_predictions.csv").write_text("")
        with self.assertRaisesRegex(ValueError, "CV|artifact"):
            c.comparison_table([experiment], stage="cv", output=self.root / "changed.csv")


if __name__ == "__main__":
    unittest.main()
