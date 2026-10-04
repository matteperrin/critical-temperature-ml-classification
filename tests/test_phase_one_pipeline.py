"""Offline acquisition and Phase I script checks; no UCI requests."""
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from src.critical_temperature import fetch_data

ROOT = Path(__file__).resolve().parents[1]


class AcquisitionTests(unittest.TestCase):
    def archive(self, files):
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        return buffer.getvalue()

    def download(self, root, files, expected):
        manifest = root / "source_manifest.json"
        manifest.write_text(json.dumps({"files": {
            name: {"sha256": sha256(content).hexdigest(), "size_bytes": len(content)}
            for name, content in expected.items()
        }}), encoding="utf-8")
        with patch.object(fetch_data, "DATA_DIR", root / "raw"), \
                patch.object(fetch_data, "MANIFEST_PATH", manifest), \
                patch.object(fetch_data, "urlopen", return_value=BytesIO(self.archive(files))):
            return fetch_data.download_data()

    def test_verified_archive_and_acquisition_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {"train.csv": b"feature,critical_temp\n1,77\n",
                     "unique_m.csv": b"material,critical_temp\nA,77\n"}
            paths = self.download(root, files, files)
            self.assertEqual([p.name for p in paths], list(fetch_data.DATA_FILES))
            for path in paths:
                self.assertEqual(path.read_bytes(), files[path.name])
            record = json.loads((root / "raw" / "acquisition.json").read_text())
            self.assertEqual(record["source_url"], fetch_data.DATASET_URL)
            self.assertTrue(record["downloaded_at_utc"].endswith("+00:00"))
            self.assertEqual(record["files"]["train.csv"]["sha256"], sha256(files["train.csv"]).hexdigest())

    def test_changed_or_missing_member_does_not_overwrite_raw(self):
        expected = {"train.csv": b"train", "unique_m.csv": b"unique"}
        for files in ({"train.csv": b"changed", "unique_m.csv": b"unique"},
                      {"train.csv": b"train"}):
            with self.subTest(files=files), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "raw").mkdir()
                for name in expected:
                    (root / "raw" / name).write_bytes(b"existing")
                with self.assertRaisesRegex((ValueError, RuntimeError), "hash|missing|identity"):
                    self.download(root, files, expected)
                for name in expected:
                    self.assertEqual((root / "raw" / name).read_bytes(), b"existing")
                self.assertFalse((root / "raw" / "acquisition.json").exists())

    def test_committed_manifest_matches_local_raw_if_present(self):
        manifest = json.loads((ROOT / "data" / "source_manifest.json").read_text())
        for name, expected in manifest["files"].items():
            path = ROOT / "data" / "raw" / name
            if path.exists():
                self.assertEqual(sha256(path.read_bytes()).hexdigest(), expected["sha256"])
                self.assertEqual(path.stat().st_size, expected["size_bytes"])


class PhaseOneScriptTests(unittest.TestCase):
    def run_script(self, name, root):
        return subprocess.run([sys.executable, str(ROOT / "src" / "critical_temperature" / name)],
                              cwd=root, capture_output=True, text=True,
                              env={**os.environ, "MPLBACKEND": "Agg"}, timeout=60)

    def inputs(self, root, invalid=False):
        (root / "data" / "raw").mkdir(parents=True)
        train = pd.DataFrame({"number_of_elements": [1, 1, 2],
                              "wtd_mean_Valence": [1.0, 1.0, 2.0],
                              "critical_temp": [77.0, 77.0, 77.01]})
        unique = pd.DataFrame({"A": [1, 1, 1], "B": [0, 0, 1],
                               "critical_temp": train.critical_temp,
                               "material": ["A", "A2", "AB"]})
        if invalid:
            train.loc[0, "critical_temp"] = -1
            unique.loc[0, "critical_temp"] = -1
        train.to_csv(root / "data" / "raw" / "train.csv", index=False)
        unique.to_csv(root / "data" / "raw" / "unique_m.csv", index=False)

    def test_cleaning_preserves_duplicates_raw_and_threshold_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.inputs(root)
            before = {p.name: p.read_bytes() for p in (root / "data" / "raw").glob("*.csv")}
            for script in ("data_cleaning.py", "data_transformation.py"):
                result = self.run_script(script, root)
                self.assertEqual(result.returncode, 0, result.stderr)
            output = pd.read_csv(root / "data" / "processed" / "train_transformed.csv")
            self.assertEqual(len(output), 3)
            self.assertEqual(output.above_77k.tolist(), [0, 0, 1])
            self.assertEqual(int(output.drop(columns="above_77k").duplicated().sum()), 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "data" / "raw").glob("*.csv")})

    def test_offline_pipeline_generates_reports_from_processed_data(self):
        from src.critical_temperature.data_analysis import run_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.inputs(root)
            for script in ("data_inspection.py", "data_cleaning.py", "data_transformation.py"):
                result = self.run_script(script, root)
                self.assertEqual(result.returncode, 0, result.stderr)
            processed = root / "data" / "processed" / "train_transformed.csv"
            before = processed.read_bytes()
            tables = run_analysis(root)
            self.assertEqual(tables["class_balance.csv"].record_count.tolist(), [2, 1])
            self.assertEqual(len(list((root / "reports" / "tables").glob("*.csv"))), 12)
            self.assertEqual(len(list((root / "reports" / "figures").glob("*.png"))), 8)
            self.assertEqual(processed.read_bytes(), before)

    def test_invalid_data_fails_before_processed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.inputs(root, invalid=True)
            result = self.run_script("data_cleaning.py", root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Data validation failed", result.stderr)
            self.assertFalse((root / "data" / "processed").exists())


if __name__ == "__main__":
    unittest.main()
