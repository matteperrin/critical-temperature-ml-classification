"""Synthetic checks for the final-holdout group-paired permutation test."""

import itertools
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.critical_temperature.statistical_comparison import paired_group_permutation


class PairedGroupPermutationTests(unittest.TestCase):
    def test_two_group_exact_result_and_metadata(self):
        result = paired_group_permutation([0, 1], [1, 0], [0, 1], ["a", "b"])
        self.assertEqual(result["difference"], 1.0)
        self.assertEqual(result["p_value"], 0.5)
        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["permutations_evaluated"], 4)
        self.assertEqual(result["evaluation_rows"], 2)
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["informative_groups"], 2)
        self.assertEqual(result["random_state"], 42)
        self.assertEqual(result["metric"], "balanced_accuracy")
        self.assertEqual(result["alternative"], "two-sided")
        self.assertTrue(result["test_name"])
        self.assertTrue(result["assumptions"])
        json.dumps(result, allow_nan=False)

    def test_duplicate_rows_are_swapped_as_one_group(self):
        grouped = paired_group_permutation([0, 0, 1, 1], [1, 1, 0, 0],
                                           [0, 0, 1, 1], ["a", "a", "b", "b"])
        independent = paired_group_permutation([0, 0, 1, 1], [1, 1, 0, 0],
                                               [0, 0, 1, 1], range(4))
        self.assertEqual(grouped["p_value"], 0.5)
        self.assertEqual(independent["p_value"], 0.125)

    def test_identical_predictions(self):
        result = paired_group_permutation([0, 1], [0, 0], [0, 0], [1, 2])
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["informative_groups"], 0)
        self.assertEqual(result["permutations_evaluated"], 1)

    def test_record_class_weights_match_sklearn(self):
        truth = [0, 0, 0, 1, 1]
        baseline = [0, 1, 1, 0, 1]
        improved = [0, 0, 1, 1, 1]
        result = paired_group_permutation(truth, baseline, improved, [0, 0, 0, 1, 2])
        self.assertAlmostEqual(result["baseline_balanced_accuracy"],
                               balanced_accuracy_score(truth, baseline))
        self.assertAlmostEqual(result["improved_balanced_accuracy"],
                               balanced_accuracy_score(truth, improved))
        self.assertAlmostEqual(result["difference"],
                               balanced_accuracy_score(truth, improved)
                               - balanced_accuracy_score(truth, baseline))

    def test_exact_distribution_matches_independent_whole_group_swaps(self):
        truth = np.array([0, 1, 1, 1, 0, 1, 1])
        baseline = np.array([0, 1, 0, 1, 1, 0, 0])
        improved = np.array([0, 0, 1, 1, 0, 1, 1])
        groups = np.array(["a", "a", "b", "b", "c", "c", "c"])
        observed = balanced_accuracy_score(truth, improved) - balanced_accuracy_score(truth, baseline)
        extremes = 0
        for swaps in itertools.product((False, True), repeat=3):
            left, right = baseline.copy(), improved.copy()
            for label, swap in zip(("a", "b", "c"), swaps):
                if swap:
                    mask = groups == label
                    left[mask], right[mask] = improved[mask], baseline[mask]
            statistic = balanced_accuracy_score(truth, right) - balanced_accuracy_score(truth, left)
            extremes += abs(statistic) >= abs(observed) - 1e-12
        result = paired_group_permutation(truth, baseline, improved, groups)
        self.assertAlmostEqual(result["difference"], observed)
        self.assertEqual(result["p_value"], extremes / 8)
        self.assertEqual(result["p_value"], 0.75)

    def test_swapping_models_preserves_p_and_negates_difference(self):
        args = ([0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1], [0, 1, 2, 3])
        forward = paired_group_permutation(*args)
        reverse = paired_group_permutation(args[0], args[2], args[1], args[3])
        self.assertEqual(forward["p_value"], reverse["p_value"])
        self.assertEqual(forward["difference"], -reverse["difference"])

    def test_exact_cancellation_removes_zero_groups(self):
        # 1/(2*3) - 3/(2*9) cancels mathematically despite float roundoff.
        truth = [0] * 3 + [1] * 9
        baseline = [1, 0, 0] + [1] * 9
        improved = [0] * 3 + [0] * 3 + [1] * 6
        groups = ["effect", "zero", "zero"] + ["effect"] * 3 + ["zero"] * 6
        result = paired_group_permutation(truth, baseline, improved, groups)
        self.assertEqual(result["informative_groups"], 0)
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["p_value"], 1.0)

    def test_mixed_string_integer_groups_do_not_merge(self):
        result = paired_group_permutation([0, 1], [1, 0], [0, 1], [1, "1"])
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["p_value"], 0.5)

    def test_exact_threshold_is_sixteen_nonzero_groups(self):
        truth = np.arange(16) % 2
        result = paired_group_permutation(truth, 1 - truth, truth, range(16),
                                          permutations=1)
        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["permutations_evaluated"], 65536)
        self.assertEqual(result["p_value"], 2 / 65536)

    def test_monte_carlo_is_seeded_and_uses_plus_one(self):
        truth = np.tile([0, 1], 17)
        args = (truth, 1 - truth, truth, np.repeat(np.arange(17), 2))
        first = paired_group_permutation(*args, permutations=999, random_state=7)
        second = paired_group_permutation(*args, permutations=999, random_state=7)
        self.assertEqual(first, second)
        self.assertEqual(first["method"], "monte_carlo")
        self.assertEqual(first["permutations_evaluated"], 999)
        self.assertEqual(first["informative_groups"], 17)
        self.assertEqual(first["p_value"], 1 / 1000)

    def test_rejects_invalid_arrays(self):
        invalid = [([], [], [], []), ([0], [0], [0], [0]),
                   ([[0, 1]], [0, 1], [0, 1], [0, 1]),
                   ([0, 1], [0], [0, 1], [0, 1]),
                   ([0, 2], [0, 1], [0, 1], [0, 1]),
                   ([0, 1], [0, np.nan], [0, 1], [0, 1]),
                   ([0, 1], [0, 1], [0, 2], [0, 1]),
                   (["0", "1"], [0, 1], [0, 1], [0, 1]),
                   ([0, pd.NA], [0, 1], [0, 1], [0, 1])]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                paired_group_permutation(*args)

    def test_rejects_missing_nonfinite_unhashable_groups(self):
        for bad in (None, np.nan, pd.NA, np.inf, -np.inf, [], {}, [1]):
            groups = np.empty(2, dtype=object)
            groups[:] = ["valid", bad]
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired_group_permutation([0, 1], [0, 1], [0, 1], groups)
        with self.assertRaises(ValueError):
            paired_group_permutation([0, 1], [0, 1], [0, 1], [[1], [2]])

    def test_rejects_invalid_controls(self):
        for value in (0, -1, 1.5, True, np.bool_(False), "10", None):
            with self.subTest(permutations=value), self.assertRaises(ValueError):
                paired_group_permutation([0, 1], [0, 1], [0, 1], [0, 1],
                                         permutations=value)
        for value in (-1, 1.5, True, np.bool_(False), "42", None):
            with self.subTest(random_state=value), self.assertRaises(ValueError):
                paired_group_permutation([0, 1], [0, 1], [0, 1], [0, 1],
                                         random_state=value)


if __name__ == "__main__":
    unittest.main()
