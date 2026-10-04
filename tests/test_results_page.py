"""Check the offline presentation against its frozen evidence, without fitting models."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from html.parser import HTMLParser

from src.critical_temperature.build_results_page import build_page

ROOT = Path(__file__).resolve().parents[1]


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.scripts = []
        self.scores = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a":
            self.links.append(attrs["href"])
        if tag == "script":
            self.scripts.append(attrs)
        if "data-score" in attrs:
            self.scores.append(float(attrs["data-score"]))


class ResultsPageTests(unittest.TestCase):
    def test_checked_in_page_is_current_and_offline(self):
        page = build_page()
        self.assertEqual((ROOT / "results.html").read_text(encoding="utf-8"), page)
        parsed = PageParser()
        parsed.feed(page)
        self.assertFalse(parsed.scripts)
        self.assertIn('name="viewport"', page)
        self.assertIn('class="skip-link"', page)
        self.assertIn("prefers-reduced-motion", page)
        self.assertIn("historical", page.lower())
        self.assertIn("4.8 percentage points lower", page)
        self.assertIn("0.00005", page)
        self.assertEqual(page.count('class="rule-example"'), 3)
        for link in parsed.links:
            if link.startswith("#"):
                self.assertIn(link[1:], parsed.ids)
            elif not link.startswith(("https:", "http:")):
                self.assertTrue((ROOT / link.split("#")[0]).is_file(), link)
        with (ROOT / "reports/holdout/final_comparison/test_comparison.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(sorted(parsed.scores), sorted(float(row["balanced_accuracy"]) * 100 for row in rows))
        for row in rows:
            self.assertIn(f'{float(row["balanced_accuracy"]) * 100:.1f}%', page)

    def test_changed_evidence_is_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "reports/holdout/final_comparison"
            shutil.copytree(ROOT / "reports/holdout/final_comparison", folder)
            with (folder / "test_comparison.csv").open("a") as file:
                file.write("changed evidence\n")
            with self.assertRaisesRegex(ValueError, "changed|hash"):
                build_page(root)
            self.assertFalse((root / "results.html").exists())

    def test_missing_member_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = Path("reports/holdout/final_comparison")
            shutil.copytree(ROOT / final, root / final)
            plan = json.loads((root / final / "plan.json").read_text())
            for entry in plan["experiments"]:
                path = Path(entry["directory"])
                shutil.copytree(ROOT / path, root / path)
            member = root / plan["primary_comparison"]["improved"] / "rules/member_73/rules.csv"
            member.unlink()
            with self.assertRaisesRegex(ValueError, "missing|changed"):
                build_page(root)


if __name__ == "__main__":
    unittest.main()
