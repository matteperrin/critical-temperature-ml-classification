"""Numeric group-paired permutation comparison for a single shared holdout."""

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


def paired_group_permutation(
    y_true, baseline_predictions, improved_predictions, groups, *,
    permutations=19999, random_state=42,
):
    """Test record-weighted balanced accuracy, improved minus baseline.

    Both models must predict the same final shared holdout rows. This is not
    a test for pooled cross-validation predictions. The null requires independent
    groups and exchangeability of the two entire prediction vectors within each
    group. Feature-vector groups do not fully identify related material families.
    Each row has fixed weight 1/(2 * its truth-class count); groups are not equally
    weighted. Zero-effect groups are removed analytically, with exact enumeration
    through 16 informative groups, otherwise seeded Monte Carlo with plus-one
    correction. No significance interpretation is returned.
    """
    for name, value, minimum in (
        ("permutations", permutations, 1), ("random_state", random_state, 0),
    ):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    permutations, random_state = int(permutations), int(random_state)

    arrays = []
    for name, values in (
        ("y_true", y_true), ("baseline_predictions", baseline_predictions),
        ("improved_predictions", improved_predictions), ("groups", groups),
    ):
        try:
            array = np.asarray(values, dtype=object)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a nonempty 1D array.") from error
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty 1D array.")
        arrays.append(array)
    truth, baseline, improved, group_labels = arrays
    if any(len(array) != len(truth) for array in arrays):
        raise ValueError("All inputs must have equal lengths.")
    for name, values in zip(
        ("y_true", "baseline_predictions", "improved_predictions"), arrays[:3],
    ):
        if any(not isinstance(value, (Real, np.bool_)) or value not in (0, 1)
               for value in values):
            raise ValueError(f"{name} must contain only binary numeric labels 0/1.")
    truth, baseline, improved = (
        values.astype(np.int8) for values in arrays[:3]
    )
    counts = np.bincount(truth, minlength=2)
    if np.any(counts == 0):
        raise ValueError("y_true must contain both classes.")

    for label in group_labels:
        try:
            hash(label)
            missing = pd.isna(label)
            if not isinstance(missing, (bool, np.bool_)):
                raise ValueError("Group IDs must be scalar labels.")
            if missing or (isinstance(label, (float, complex, np.inexact))
                           and not np.isfinite(label)):
                raise ValueError("Group IDs must be finite and non-missing.")
        except (TypeError, ValueError) as error:
            raise ValueError("Group IDs must be finite, non-missing hashable labels.") from error
    codes, unique_groups = pd.factorize(group_labels, sort=False)
    group_count = len(unique_groups)
    baseline_correct = baseline == truth
    improved_correct = improved == truth
    correctness_delta = improved_correct.astype(np.int64) - baseline_correct

    # Integer class counts identify true cancellation without rounding small
    # nonzero effects to zero: d_g = (delta0*N1 + delta1*N0)/(2*N0*N1).
    class_deltas = np.zeros((group_count, 2), dtype=np.int64)
    np.add.at(class_deltas, (codes, truth), correctness_delta)
    n0, n1 = map(int, counts)
    denominator = 2 * n0 * n1
    numerators = [int(delta0) * n1 + int(delta1) * n0
                  for delta0, delta1 in class_deltas]
    effects = np.array([value / denominator for value in numerators if value != 0])
    informative_groups = len(effects)
    difference = sum(numerators) / denominator
    observed = abs(difference)
    # Scale the equality tolerance to the attainable statistics, not to 1.0.
    tolerance = 8 * np.finfo(float).eps * math.fsum(abs(value) for value in effects)
    chunk_size = max(1, min(4096, 262144 // max(1, informative_groups)))
    exceedances = 0
    if informative_groups <= 16:
        method = "exact"
        draws = 1 << informative_groups
        for start in range(0, draws, chunk_size):
            states = np.arange(start, min(start + chunk_size, draws), dtype=np.uint32)
            bits = (states[:, None] >> np.arange(informative_groups, dtype=np.uint32)) & 1
            signs = bits.astype(np.int8) * 2 - 1
            statistics = np.sum(signs * effects, axis=1)
            exceedances += int(np.count_nonzero(np.abs(statistics) >= observed - tolerance))
        p_value = exceedances / draws
    else:
        method = "monte_carlo"
        draws = permutations
        rng = np.random.default_rng(random_state)
        for start in range(0, draws, chunk_size):
            signs = rng.integers(0, 2, size=(min(chunk_size, draws - start),
                                            informative_groups), dtype=np.int8) * 2 - 1
            statistics = np.sum(signs * effects, axis=1)
            exceedances += int(np.count_nonzero(np.abs(statistics) >= observed - tolerance))
        p_value = (exceedances + 1) / (draws + 1)

    def balanced_accuracy(correct):
        return float(sum(np.count_nonzero(correct & (truth == label)) / int(counts[label])
                         for label in (0, 1)) / 2)

    return {
        "baseline_balanced_accuracy": balanced_accuracy(baseline_correct),
        "improved_balanced_accuracy": balanced_accuracy(improved_correct),
        "difference": difference,
        "p_value": p_value,
        "method": method,
        "evaluation_rows": len(truth),
        "group_count": group_count,
        "informative_groups": informative_groups,
        "permutations_evaluated": draws,
        "random_state": random_state,
        "seed": random_state,
        "draws": draws,
        "alpha": 0.05,
        "test_name": "paired_group_permutation",
        "metric": "balanced_accuracy",
        "alternative": "two-sided",
        "assumptions": [
            "Both prediction vectors refer to the same final shared holdout rows, not pooled cross-validation.",
            "Groups are independent under the null.",
            "Entire model prediction vectors are exchangeable within each group under the null.",
            "Feature-vector groups do not fully identify related material families.",
        ],
    }
