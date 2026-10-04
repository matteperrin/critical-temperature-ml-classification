"""Exploratory Phase I regeneration keeps paired records and source bytes intact."""
import os
os.environ["MPLBACKEND"] = "Agg"

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.critical_temperature import data_analysis
from src.critical_temperature import phase_one_reports


TABLES = {
    "class_balance.csv", "cleaning_summary.csv", "data_quality_summary.csv",
    "descriptive_statistics.csv", "feature_target_correlations.csv",
    "repeated_formula_examples.csv", "top_iqr_outliers.csv", "near_zero_variance.csv",
    "class_feature_summary.csv", "composition_complexity.csv",
    "element_prevalence.csv", "composition_frequency.csv",
}
FIGURES = {
    "class_distribution.png", "critical_temperature_histogram.png",
    "critical_temperature_boxplot.png", "correlation_heatmap.png",
    "weighted_mean_valence_scatter.png", "class_feature_distribution.png",
    "composition_complexity_temperature.png", "element_prevalence.png",
}


class PhaseOneReportsTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        temperature = np.r_[np.repeat(77., 20), 78., 1000.]
        train = pd.DataFrame({
            "number_of_elements": np.repeat(2, 22),
            "wtd_mean_Valence": -temperature,
            "positive": temperature * 2,
            "constant": np.ones(22),
            "rare": np.r_[np.zeros(21), 1],
            "critical_temp": temperature,
        })
        unique = pd.DataFrame({"H": np.ones(22), "O": np.ones(22),
                               "He": np.zeros(22), "critical_temp": temperature,
                               "material": np.repeat("H1O1", 22)})
        train.to_csv(raw / "train.csv", index=False)
        unique.to_csv(raw / "unique_m.csv", index=False)
        return train, unique

    def test_regeneration_signed_correlations_nzv_boundary_and_readonly_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            originals = {p: p.read_bytes() for p in (root / "data/raw").glob("*.csv")}
            tables = phase_one_reports.generate_reports(root)
            self.assertEqual(set(tables), TABLES)
            self.assertEqual({p.name for p in (root / "reports/tables").glob("*.csv")}, TABLES)
            correlations = tables["feature_target_correlations.csv"].set_index("feature")
            self.assertNotIn("above_77k", correlations.index)
            self.assertNotIn("critical_temp", correlations.index)
            self.assertAlmostEqual(correlations.loc["wtd_mean_Valence", "pearson_correlation"], -1)
            self.assertAlmostEqual(correlations.loc["wtd_mean_Valence", "spearman_correlation"], -1)
            nzv = tables["near_zero_variance.csv"].set_index("feature")
            self.assertTrue(nzv.loc["constant", "constant"])
            self.assertTrue(nzv.loc["rare", "near_zero_variance"])
            self.assertFalse(nzv.loc["positive", "near_zero_variance"])
            balance = tables["class_balance.csv"].set_index("above_77k")
            self.assertEqual(balance.record_count.to_dict(), {0: 20, 1: 2})
            frequency = tables["composition_frequency.csv"]
            self.assertEqual(frequency.record_count.sum(), 22)
            self.assertTrue(frequency.crosses_77k_boundary.iloc[0])
            prevalence = tables["element_prevalence.csv"].set_index("element")
            self.assertEqual(prevalence.loc["H", "record_count"], 22)
            self.assertEqual(prevalence.loc["He", "record_count"], 0)
            self.assertEqual(tables["composition_complexity.csv"].record_count.sum(), 22)
            summary = tables["class_feature_summary.csv"]
            self.assertEqual(set(summary.above_77k), {0, 1})
            data_analysis.run_analysis(root)
            self.assertEqual({p.name for p in (root / "reports/figures").glob("*.png")}, FIGURES)
            for p, original in originals.items():
                self.assertEqual(p.read_bytes(), original)

    def test_nzv_strict_frequency_and_unique_percentage_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 40
            train = pd.DataFrame({
                "number_of_elements": np.ones(n), "critical_temp": np.arange(n) + 1.,
                "ratio_19": np.r_[np.zeros(38), np.ones(2)],
                "ratio_39": np.r_[np.zeros(39), 1],
                "unique_10_percent": np.r_[np.zeros(37), 1, 2, 3],
                "unique_over_10_percent": np.r_[np.zeros(36), 1, 2, 3, 4],
            })
            unique = pd.DataFrame({"H": np.ones(n), "material": ["H"] * n,
                                   "critical_temp": train.critical_temp})
            tables = phase_one_reports.generate_reports(tmp, train=train, unique=unique)
            nzv = tables["near_zero_variance.csv"].set_index("feature")
            self.assertFalse(nzv.loc["ratio_19", "near_zero_variance"])
            self.assertTrue(nzv.loc["ratio_39", "near_zero_variance"])
            self.assertTrue(nzv.loc["unique_10_percent", "near_zero_variance"])
            self.assertFalse(nzv.loc["unique_over_10_percent", "near_zero_variance"])

    def test_misaligned_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, unique = self.fixture(root)
            unique.loc[0, "critical_temp"] = 76
            with self.assertRaisesRegex(ValueError, "align"):
                phase_one_reports.generate_reports(root, train=train, unique=unique)


if __name__ == "__main__":
    unittest.main()
