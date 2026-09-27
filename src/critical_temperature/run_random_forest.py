
"""Evaluate Random Forest using the original superconductivity dataset."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold

if __package__:
    from .model_data import load_model_data
else:
    from model_data import load_model_data


RANDOM_STATE = 42
N_SPLITS = 5
N_TREES = 200

REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports/random_forest_raw.csv"
)

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


def main():
    # Load the same raw dataset used by eLCS.
    X, y, groups = load_model_data()

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        # Train a fresh Random Forest for every fold.
        model = RandomForestClassifier(
            n_estimators=N_TREES,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        scores = {
            "model": "Random Forest",
            "data": "data/raw/train.csv",
            "validation": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "n_estimators": N_TREES,
            "fold": fold,
            "training_rows": len(train_idx),
            "test_rows": len(test_idx),
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test, probabilities
            ),
            "pr_auc": average_precision_score(
                y_test, probabilities
            ),
        }

        results.append(scores)

        print(
            f"Fold {fold}: "
            f"Balanced accuracy = "
            f"{scores['balanced_accuracy']:.3f}"
        )

    # Calculate summary statistics.
    fold_results = pd.DataFrame(results)

    metadata = {
        "model": "Random Forest",
        "data": "data/raw/train.csv",
        "validation": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "n_estimators": N_TREES,
    }

    summary = pd.DataFrame([
        {
            **metadata,
            "fold": statistic,
            **fold_results[METRICS].agg(statistic).to_dict(),
        }
        for statistic in ("mean", "std")
    ])

    output = pd.concat(
        [fold_results, summary],
        ignore_index=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        REPORT_PATH,
        index=False,
        float_format="%.6f",
    )

    print(f"\nResults saved to: {REPORT_PATH}")
    print(
        "Mean balanced accuracy: "
        f"{fold_results['balanced_accuracy'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
