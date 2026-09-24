"""Evaluate the unmodified eLCS baseline on the raw dataset."""

from pathlib import Path

import pandas as pd
from skeLCS import eLCS
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

if __package__:
    from .model_data import load_model_data
else:
    from model_data import load_model_data

RANDOM_STATE = 42
LEARNING_ITERATIONS = 100
POPULATION_SIZE = 100
N_SPLITS = 5
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports/elcs_raw_baseline.csv"
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1")


def main() -> None:
    """Evaluate unmodified eLCS across all grouped stratified folds."""
    X, y, groups = load_model_data()
    X_values = X.to_numpy()
    y_values = y.to_numpy()
    group_values = groups.to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    results = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(X_values, y_values, groups=group_values), start=1
    ):
        model = eLCS(
            learning_iterations=LEARNING_ITERATIONS,
            N=POPULATION_SIZE,
            random_state=RANDOM_STATE,
        )
        model.fit(X_values[train_indices], y_values[train_indices])
        predictions = model.predict(X_values[test_indices])
        results.append(
            {
                "model": "unmodified eLCS",
                "data": "data/raw/train.csv",
                "validation": "StratifiedGroupKFold",
                "n_splits": N_SPLITS,
                "random_state": RANDOM_STATE,
                "learning_iterations": LEARNING_ITERATIONS,
                "population_size": POPULATION_SIZE,
                "fold": fold,
                "training_rows": len(train_indices),
                "test_rows": len(test_indices),
                "accuracy": accuracy_score(y_values[test_indices], predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    y_values[test_indices], predictions
                ),
                "precision": precision_score(
                    y_values[test_indices], predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_values[test_indices], predictions, zero_division=0
                ),
                "f1": f1_score(y_values[test_indices], predictions, zero_division=0),
            }
        )

    fold_results = pd.DataFrame(results)
    metadata = {
        "model": "unmodified eLCS",
        "data": "data/raw/train.csv",
        "validation": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "learning_iterations": LEARNING_ITERATIONS,
        "population_size": POPULATION_SIZE,
    }
    summary = pd.DataFrame(
        [
            {
                **metadata,
                "fold": statistic,
                **fold_results[list(METRICS)].agg(statistic).to_dict(),
            }
            for statistic in ("mean", "std")
        ]
    )
    output = pd.concat([fold_results, summary], ignore_index=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(REPORT_PATH, index=False, float_format="%.6f")

    print(f"Saved {N_SPLITS}-fold baseline results to {REPORT_PATH}")
    print(
        f"Balanced accuracy: {fold_results['balanced_accuracy'].mean():.3f} "
        f"(+/- {fold_results['balanced_accuracy'].std():.3f})"
    )


if __name__ == "__main__":
    main()
