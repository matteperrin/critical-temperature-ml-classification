"""Offline presentation of the completed exploratory revision and its evidence."""
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.critical_temperature.build_results_page import ROOT, build_page

REVISION = Path("reports/holdout/elcs_exploratory_revision_v2")


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.scripts, self.scores = [], set(), [], []
        self.references, self.budgets = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))
        if tag == "script":
            self.scripts.append(attrs)
        if "data-score" in attrs:
            self.scores.append(float(attrs["data-score"]))
        if attrs.get("id") == "majority-reference":
            self.references.append(attrs)
        if attrs.get("id") == "learning-budget":
            self.budgets.append(attrs)


class ResultsPageTests(unittest.TestCase):
    def copy_evidence(self, root):
        shutil.copytree(ROOT / REVISION, root / REVISION)
        conventional = Path("reports/holdout/conventional_defaults")
        shutil.copytree(ROOT / conventional, root / conventional)

    def test_checked_in_page_is_current_and_offline(self):
        page = build_page()
        self.assertEqual(page, (ROOT / "results.html").read_text(encoding="utf-8"))
        parsed = PageParser()
        parsed.feed(page)
        self.assertFalse(parsed.scripts)
        self.assertEqual(len(parsed.scores), 5)
        self.assertEqual(parsed.scores, sorted(parsed.scores, reverse=True))
        for href in parsed.links:
            if href.startswith("#"):
                self.assertIn(href[1:], parsed.ids)
            elif not href.startswith(("https://", "http://")):
                self.assertTrue((ROOT / href.split("#")[0]).is_file(), href)
        self.assertNotIn("<link", page)
        self.assertNotIn("@import", page)
        self.assertNotIn("url(", page)
        self.assertIn("prefers-reduced-motion", page)
        self.assertIn("@media print", page)
        with (ROOT / REVISION / "descriptive_test_table.csv").open(newline="") as file:
            for row in csv.DictReader(file):
                self.assertIn(f'{float(row["balanced_accuracy"]) * 100:.1f}%', page)

    def test_revision_budget_reference_and_claims_match_saved_evidence(self):
        page = build_page()
        parsed = PageParser()
        parsed.feed(page)
        selection = json.loads((ROOT / REVISION / "selection.json").read_text())
        folder = ROOT / REVISION / selection["selected_single_model"]["experiment"]
        with (folder / "test_results.csv").open(newline="") as file:
            result = next(csv.DictReader(file))
        iterations = selection["selected_single_model"]["iterations_per_member"]
        training_rows = int(result["training_rows"])
        self.assertEqual(len(parsed.references), 1)
        self.assertEqual(len(parsed.budgets), 1)
        self.assertEqual(int(parsed.budgets[0]["data-learning-iterations"]), iterations)
        self.assertEqual(int(parsed.budgets[0]["data-training-rows"]), training_rows)
        self.assertEqual(float(parsed.budgets[0]["data-learning-passes"]), iterations / training_rows)
        negatives = int(result["true_negatives"]) + int(result["false_positives"])
        self.assertAlmostEqual(float(parsed.references[0]["data-reference-accuracy"]), negatives / int(result["evaluation_rows"]))
        self.assertEqual(float(parsed.references[0]["data-reference-balanced-accuracy"]), 0.5)
        self.assertEqual(float(parsed.references[0]["data-reference-recall"]), 0)
        self.assertIn("Exploratory revision", page)
        self.assertIn("reused evaluation records", page)
        self.assertIn("0.54 percentage points higher", page)
        self.assertIn("No new significance test", page)
        self.assertIn("339,280", page)
        self.assertIn("20 complete passes", page)
        self.assertIn("5.9%", page)
        self.assertNotIn("statistical_test.json", page)
        self.assertNotIn("p = 0.00005", page)
        self.assertNotIn("The ensemble did not help", page)
        self.assertNotIn("elcs_development_v1/", page)
        self.assertIn("Repeated presentations", page)
        self.assertIn("Only unspecified features are unrestricted", page)

    def test_conclusions_and_commit_based_process_are_present(self):
        page = build_page()
        parsed = PageParser()
        parsed.feed(page)
        self.assertIn("conclusions", parsed.ids)
        self.assertIn("process", parsed.ids)
        self.assertIn("Our process", page)
        self.assertIn("Precision is not candidate coverage", page)
        self.assertIn("Development and evaluation tell different stories", page)
        self.assertIn("Historical comparisons", page)
        self.assertIn("639a5fc", page)
        self.assertIn("4d94af2", page)
        self.assertIn("4d50ece", page)
        self.assertIn("not a complete record of individual contributions", page)

    def test_changed_evidence_is_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_evidence(root)
            with (root / REVISION / "descriptive_test_table.csv").open("a") as file:
                file.write("changed evidence\n")
            with self.assertRaisesRegex(ValueError, "changed|hash"):
                build_page(root)
            self.assertFalse((root / "results.html").exists())

    def test_changed_cv_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_evidence(root)
            selection = json.loads((root / REVISION / "selection.json").read_text())
            results = root / REVISION / selection["selected_single_model"]["experiment"] / "cv_results.csv"
            results.write_text("changed training counts\n")
            with self.assertRaisesRegex(ValueError, "changed|hash"):
                build_page(root)

    def test_missing_member_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_evidence(root)
            (root / REVISION / "corrected_ensemble_dedup_selected/rules/member_73/rules.csv").unlink()
            with self.assertRaisesRegex(ValueError, "missing|changed"):
                build_page(root)

    def test_changed_conventional_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_evidence(root)
            (root / "reports/holdout/conventional_defaults/random_forest/test_predictions.csv").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "changed|hash"):
                build_page(root)

    def test_incomplete_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_evidence(root)
            (root / REVISION / "study_complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing|complete"):
                build_page(root)


if __name__ == "__main__":
    unittest.main()
