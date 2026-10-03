"""Bundled runtime isolation and frozen provenance, without expensive training."""
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src.critical_temperature import run_holdout as h


class RuntimeTests(unittest.TestCase):
    def test_configuration_api(self):
        config = h.model_configuration("ensemble_dedup", 7, [11, 42, 73],
                                       population_size=23, library_variant="unmodified")
        self.assertEqual(config["parameters"]["N"], 23)
        self.assertEqual(config["library_variant"], "unmodified")
        self.assertEqual(config["runtime"]["package_version"], "1.2.4")

    def test_isolated_variants_and_training_only_majority(self):
        from src.critical_temperature import elcs_runtime as r
        sentinel = object()
        with patch.dict(sys.modules, {"skeLCS": sentinel}):
            corrected = r.create_model({"learning_iterations": 1, "N": 10, "random_state": 42}, "corrected")
            baseline = r.create_model({"learning_iterations": 1, "N": 10, "random_state": 42}, "unmodified")
            X = np.array([[0.], [1.], [2.], [3.]])
            y = np.array([0., 0., 0., 1.])
            corrected.fit(X, y)
            baseline.fit(X, y)
            self.assertEqual(corrected.env.formatData.majorityClass, 0)
            self.assertEqual(baseline.env.formatData.majorityClass, 1)
            self.assertIs(sys.modules["skeLCS"], sentinel)
            self.assertNotEqual(type(corrected).__module__, type(baseline).__module__)
            self.assertFalse(any(name.startswith("_holdout_elcs_") for name in sys.modules))
            for model, expected in ((corrected, 0), (baseline, 1)):
                # Empty population guarantees the fallback for unseen validation features.
                model.population.popSet = []
                np.testing.assert_array_equal(model.predict(np.array([[99.], [100.]])), [expected, expected])

    def test_missing_changed_provenance_and_unknown_patch_rejected(self):
        from src.critical_temperature import elcs_runtime as r
        config = h.model_configuration("elcs", 1, [11, 42, 73])
        r.validate_configuration(config)
        missing = {k: v for k, v in config.items() if k != "runtime"}
        with self.assertRaisesRegex(ValueError, "provenance"):
            r.validate_configuration(missing)
        changed = {**config, "runtime": {**config["runtime"], "package_version": "old"}}
        with self.assertRaisesRegex(ValueError, "changed"):
            r.validate_configuration(changed)
        with self.assertRaisesRegex(ValueError, "majority"):
            r.effective_sources({"DataManagement.py": b"unknown"}, "unmodified")

    def test_source_hashes_and_changed_source_rejected(self):
        from src.critical_temperature import elcs_runtime as r
        corrected_sources, corrected = r.snapshot("corrected")
        baseline_sources, baseline = r.snapshot("unmodified")
        self.assertEqual(corrected["source_sha256"], baseline["source_sha256"])
        changed = [name for name in corrected_sources if corrected_sources[name] != baseline_sources[name]]
        self.assertEqual(changed, ["DataManagement.py"])
        for name, digest in corrected["source_sha256"].items():
            self.assertEqual(digest, h.data_hash(r.BUNDLE / "skeLCS" / name))
        config = h.model_configuration("elcs", 1, [11, 42, 73])
        changed_sources = dict(corrected_sources)
        changed_sources["Timer.py"] += b"\n# changed runtime\n"
        read_bytes = type(r.BUNDLE).read_bytes

        def changed_read(path):
            if path == r.BUNDLE / "skeLCS" / "Timer.py":
                return changed_sources["Timer.py"]
            return read_bytes(path)

        with patch.object(type(r.BUNDLE), "read_bytes", changed_read):
            with self.assertRaisesRegex(ValueError, "changed"):
                r.validate_configuration(config)
            with self.assertRaisesRegex(ValueError, "changed"):
                r.create_model(config["parameters"], expected=config["runtime"])

    def test_positive_population_and_known_variant(self):
        for population in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                h.model_configuration("elcs", 1, [11, 42, 73], population_size=population)
        with self.assertRaises(ValueError):
            h.model_configuration("elcs", 1, [11, 42, 73], library_variant="unknown")


if __name__ == "__main__":
    unittest.main()
