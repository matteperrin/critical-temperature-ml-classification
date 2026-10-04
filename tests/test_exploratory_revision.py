"""Fast, mocked checks: no real model fitting or reused-test scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.critical_temperature import run_exploratory_revision as revision


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data.csv"
        self.data.write_text("data", encoding="utf-8")
        self.output = self.root / "revision"
        self.references = tuple(self.root / f"original_{i}" for i in range(3))
        for reference in self.references:
            reference.mkdir()
            (reference / "config.json").write_text("original", encoding="utf-8")
        self.originals = {str(p): p.read_bytes() for r in self.references for p in r.iterdir()}
        self.calls = []
        self.sources = {"runner.py": "frozen"}
        source = pd.DataFrame({"x": [0, 0, 1, 2, 3, 4, 5, 6]})
        self.loaded = (source, pd.Series([0, 0, 1, 0, 1, 0, 1, 0]), pd.Series(range(8)), source)
        self.splits = (np.arange(6), np.arange(6, 8), [
            (np.array([2, 3, 4, 5]), np.array([0, 1])),
            (np.array([0, 1, 4, 5]), np.array([2, 3])),
            (np.array([0, 1, 2, 3]), np.array([4, 5]))])

    def fake_cv(self, model, output, **kwargs):
        plan = json.loads((self.output / "plan.json").read_text())
        self.assertTrue(plan["exploratory"])
        self.assertIn("reused", plan["test_exposure"])
        self.assertIn("authorized", plan["authorization"])
        self.assertEqual(plan["single_seed"], 42)
        self.assertEqual(plan["ensemble_seeds"], [11, 42, 73])
        self.assertEqual(plan["training_counts"]["final_deduplicated"], 5)
        self.assertEqual(plan["training_counts"]["cv_deduplicated"], [4, 3, 3])
        self.assertEqual(json.loads((self.output / "status.json").read_text())["state"], "active")
        self.calls.append((model, output, kwargs))
        output.mkdir()
        (output / "config.json").write_text(json.dumps(kwargs, default=str))
        (output / "cv_complete.json").write_text("{}")
        value = .8 if kwargs["iterations"] >= 25 else .7
        pd.DataFrame([{"fold": "mean", **{k: value for k in
                     ("accuracy", "balanced_accuracy", "precision", "recall", "f1")}},
                      {"fold": "std", "balanced_accuracy": .01}]).to_csv(output / "cv_results.csv", index=False)

    def fake_test(self, output):
        record = json.loads((self.output / "reporting_record.json").read_text())
        self.assertTrue(record["reused_test_exploratory"])
        self.assertEqual(record["test_order"], [self.calls[2][1].name, self.calls[-1][1].name])
        for name, artifacts in record["selected_cv_artifacts_sha256"].items():
            for relative, digest in artifacts.items():
                self.assertEqual(revision.data_hash(self.output / name / relative), digest)
        (output / "test_started.json").write_text("{}")
        (output / "test_complete.json").write_text("{}")

    def fake_table(self, experiments, *, stage, output):
        self.assertEqual(len(experiments), 5)
        self.assertEqual(tuple(experiments[2:]), self.references)
        frame = pd.DataFrame({"experiment": [str(p) for p in experiments]})
        frame.to_csv(output, index=False)
        return frame

    def run_mocked(self, cv=None, fingerprint=None, comparison=None):
        with (patch.object(revision, "REFERENCES", self.references),
              patch.object(revision, "load_model_data", return_value=self.loaded),
              patch.object(revision, "create_splits", return_value=self.splits),
              patch.object(revision, "source_fingerprint", side_effect=fingerprint or (lambda: self.sources.copy())),
              patch.object(revision, "run_cv", side_effect=cv or self.fake_cv),
              patch.object(revision, "run_test", side_effect=self.fake_test) as scorer,
              patch.object(revision, "comparison_table", side_effect=comparison or self.fake_table)):
            result = revision.run_study(self.output, data_path=self.data)
            self.assertEqual(scorer.call_count, 2)
            return result

    def test_counts_grid_selection_reporting_and_preservation(self):
        self.run_mocked()
        self.assertEqual(len(self.calls), 7)
        self.assertEqual([(c[2]["iterations"], c[2]["population_size"]) for c in self.calls[:6]],
                         [(5, 100), (5, 1000), (25, 100), (25, 1000), (100, 100), (100, 1000)])
        self.assertTrue(all(c[2]["library_variant"] == "corrected" for c in self.calls))
        self.assertEqual(self.calls[-1][0], "ensemble_dedup")
        self.assertEqual(self.calls[-1][2]["iterations"], 25)
        plan = json.loads((self.output / "plan.json").read_text())
        self.assertEqual(plan["budgets"][0]["cv_pass_equivalents"], [1.25, 5 / 3, 5 / 3])
        complete = json.loads((self.output / "study_complete.json").read_text())
        for relative, digest in complete["artifacts_sha256"].items():
            self.assertEqual(revision.data_hash(self.output / relative), digest)
        self.assertEqual(json.loads((self.output / "status.json").read_text())["state"], "completed")
        self.assertTrue(all(s["elapsed_wall_seconds"] >= 0 for s in complete["completed"]))
        table = pd.read_csv(self.output / "descriptive_test_table.csv")
        self.assertTrue(table.reused_test_exploratory.all())
        self.assertEqual(table.evidence_role.tolist(), ["exploratory_revision"] * 2 + ["fixed_historical_reference"] * 3)
        for path, content in self.originals.items():
            self.assertEqual(Path(path).read_bytes(), content)

    def test_exact_ties_and_small_differences(self):
        rows = [{"balanced_accuracy": .7, "iterations_per_member": i, "population_size_per_member": n}
                for i, n in [(100, 100), (5, 1000), (5, 100)]]
        self.assertEqual(revision.select_candidate(rows), rows[-1])
        rows[0]["balanced_accuracy"] += 1e-14
        self.assertEqual(revision.select_candidate(rows), rows[0])

    def test_full_coverage_is_required(self):
        self.splits[2][-1] = (self.splits[2][-1][0], np.array([3, 4]))
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_mocked()
        self.assertFalse(self.calls)

    def test_existing_output_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_mocked()
        self.assertFalse(self.calls)

    def test_source_change_before_fit(self):
        calls = 0
        def fingerprint():
            nonlocal calls
            calls += 1
            return self.sources if calls == 1 else {"runner.py": "changed"}
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            self.run_mocked(fingerprint=fingerprint)
        self.assertFalse(self.calls)
        self.assertFalse((self.output / "study_complete.json").exists())

    def test_changes_during_fit_rejected(self):
        for kind in ("data", "source", "reference"):
            with self.subTest(kind=kind):
                self.output = self.root / kind
                self.sources = {"runner.py": "frozen"}
                def changed(model, output, **kwargs):
                    self.fake_cv(model, output, **kwargs)
                    if kind == "data":
                        self.data.write_text("changed")
                    elif kind == "source":
                        self.sources["runner.py"] = "changed"
                    else:
                        (self.references[0] / "config.json").write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    self.run_mocked(cv=changed)
                self.assertFalse((self.output / "study_complete.json").exists())
                self.assertEqual(json.loads((self.output / "status.json").read_text())["state"], "failed_or_interrupted")

    def test_comparison_failure_does_not_claim_completion(self):
        def incompatible(*args, **kwargs):
            raise ValueError("incompatible conventional reference")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.run_mocked(comparison=incompatible)
        self.assertFalse((self.output / "study_complete.json").exists())
        self.assertFalse(list(self.output.rglob("test_started.json")))

    def test_failed_test_preserves_once_only_marker_and_reporting_record(self):
        def failed_test(experiment):
            self.assertTrue((self.output / "reporting_record.json").exists())
            (experiment / "test_started.json").write_text("started")
            raise RuntimeError("final fit failed")
        with patch.object(self, "fake_test", side_effect=failed_test):
            with self.assertRaisesRegex(RuntimeError, "final fit failed"):
                self.run_mocked()
        self.assertEqual(len(list(self.output.rglob("test_started.json"))), 1)
        self.assertFalse((self.output / "study_complete.json").exists())
        self.assertEqual(json.loads((self.output / "status.json").read_text())["state"], "failed_or_interrupted")
        with self.assertRaises(FileExistsError):
            self.run_mocked()

    def test_output_inside_conventional_reference_is_rejected(self):
        self.output = self.references[0] / "nested"
        with self.assertRaisesRegex(ValueError, "inside an original"):
            self.run_mocked()
        self.assertFalse(self.output.exists())

    def test_interruption_retains_marker_and_never_completes_or_resumes(self):
        def failed(model, output, **kwargs):
            output.mkdir()
            (output / "marker").write_text("started")
            raise KeyboardInterrupt("interrupted fit")
        with self.assertRaises(KeyboardInterrupt):
            self.run_mocked(cv=failed)
        self.assertTrue(list(self.output.rglob("marker")))
        self.assertFalse((self.output / "study_complete.json").exists())
        with self.assertRaises(FileExistsError):
            self.run_mocked()


if __name__ == "__main__":
    unittest.main()
