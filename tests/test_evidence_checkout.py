"""Frozen evidence must survive Git checkout with Linux-style defaults."""
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from zipfile import ZipFile

from src.critical_temperature.build_results_page import ROOT, build_page


class EvidenceCheckoutTests(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.check_output(
            ["git", "-c", "core.autocrlf=false", "-C", str(root), *args],
            stderr=subprocess.PIPE,
        )

    def test_frozen_evidence_hashes_survive_lf_git_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp) / "repository"
            repository.mkdir()
            shutil.copytree(ROOT / "reports/holdout", repository / "reports/holdout")
            attributes = ROOT / ".gitattributes"
            if attributes.exists():
                shutil.copyfile(attributes, repository / ".gitattributes")
            self.git(repository, "init", "--quiet")
            # Reproduce the original Windows clean filter when storing the
            # evidence, then archive with Linux-style core.autocrlf=false.
            self.git(repository, "-c", "core.autocrlf=true", "add", "--all")
            tree = self.git(repository, "write-tree").decode().strip()
            archive = self.git(repository, "archive", "--format=zip", tree)
            checkout = Path(tmp) / "checkout"
            with ZipFile(BytesIO(archive)) as zip_file:
                zip_file.extractall(checkout)
            # Validate every frozen hash used by the real offline page, not a
            # synthetic manifest regenerated after Git transformed the files.
            self.assertEqual(build_page(checkout), build_page())
            for original in (ROOT / "reports/holdout").rglob("rules.csv"):
                self.assertEqual(original.read_bytes(),
                                 (checkout / original.relative_to(ROOT)).read_bytes())


if __name__ == "__main__":
    unittest.main()
