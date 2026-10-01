"""Evaluate Logistic Regression using the original superconductivity dataset."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler

if __package__:
    from .model_data import load_model_data
else:
    from model_data import load_model_data


RANDOM_STATE = 42
N_SPLITS = 5


REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports/logistic_regression_raw.csv"
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
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        scores = {
            "model": "Logistic Regression",
            "data": "data/raw/train.csv",
            "validation": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "max_iter": 1000,
            "fold": fold,
            "training_rows": len(train_idx),
            "test_rows": len(test_idx),
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_test,
                probabilities,
            ),
        
            "pr_auc": average_precision_score(
                y_test,
                probabilities,
            ),
        }

        results.append(scores)

        print(
            f"Fold {fold}: "
            f"Balanced accuracy = "
            f"{scores['balanced_accuracy']:.3f}"
        )

    fold_results = pd.DataFrame(results)

    metadata = {
        "model": "Logistic Regression",
        "data": "data/raw/train.csv",
        "validation": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }

    summary = pd.DataFrame([
        {
            **metadata,
            "fold": statistic,
            **fold_results[METRICS]
            .agg(statistic)
            .to_dict(),
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