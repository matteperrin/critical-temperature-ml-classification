"""Evaluate eLCS on raw holdouts, optionally deduplicating training rows."""

import argparse
from pathlib import Path

import pandas as pd
from skeLCS import eLCS
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
if __package__:
    from .model_data import load_model_data, model_folds
else:
    from model_data import load_model_data, model_folds


RANDOM_STATE = 42
LEARNING_ITERATIONS = 1000
POPULATION_SIZE = 100
N_SPLITS = 5

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)


def main(argv: list[str] | None = None) -> None:
    """Evaluate unmodified eLCS across all grouped stratified folds."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--iterations",
        type=int,
        default=LEARNING_ITERATIONS,
        help="Positive learning-iteration budget per fold "
        "(default: %(default)s).",
    )

    parser.add_argument(
        "--data",
        choices=("raw", "preprocessed"),
        default="raw",
        help="raw: unchanged training; preprocessed: deduplicate training rows "
        "only. Both use unchanged raw holdouts (default: %(default)s).",
    )

    args = parser.parse_args(argv)

    if args.iterations <= 0:
        parser.error("--iterations must be a positive integer")

    iterations = args.iterations

    # Always split raw rows; the global preprocessed export changes holdouts.
    source = None
    if args.data == "preprocessed":
        X, y, groups, source = load_model_data(include_source=True)
        report_path = REPORT_DIR / f"elcs_training_dedup_{iterations}_iterations.csv"
    else:
        X, y, groups = load_model_data()
        report_path = REPORT_DIR / f"elcs_raw_{iterations}_iterations.csv"

    X_values = X.to_numpy()
    y_values = y.to_numpy()
    results = []

    for fold, (train_indices, test_indices) in enumerate(
        model_folds(X, y, groups, n_splits=N_SPLITS, random_state=RANDOM_STATE),
        start=1,
    ):
        if source is not None:
            # Include continuous temperature: equal binary labels are not duplicates.
            keep = ~source.iloc[train_indices].duplicated().to_numpy()
            train_indices = train_indices[keep]
        # Create a fresh unmodified eLCS model for each fold.
        model = eLCS(
            learning_iterations=iterations,
            N=POPULATION_SIZE,
            random_state=RANDOM_STATE,
        )

        # Train only on the training portion of the fold.
        model.fit(
            X_values[train_indices],
            y_values[train_indices],
        )

        # Predict on the unseen test portion.
        predictions = model.predict(
            X_values[test_indices]
        )

        # Calculate the confusion matrix.
        tn, fp, fn, tp = confusion_matrix(
            y_values[test_indices],
            predictions,
            labels=[0, 1],
        ).ravel()

        # Store all fold-level results.
        results.append(
            {
                "model": "unmodified eLCS",
                "data": "data/raw/train.csv",
                "preprocessing": "training_only_exact_dedup" if source is not None else "none",
                "validation": "StratifiedGroupKFold",
                "n_splits": N_SPLITS,
                "random_state": RANDOM_STATE,
                "learning_iterations": iterations,
                "population_size": POPULATION_SIZE,
                "fold": fold,
                "training_rows": len(train_indices),
                "test_rows": len(test_indices),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "accuracy": accuracy_score(
                    y_values[test_indices],
                    predictions,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_values[test_indices],
                    predictions,
                ),
                "precision": precision_score(
                    y_values[test_indices],
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_values[test_indices],
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_values[test_indices],
                    predictions,
                    zero_division=0,
                ),
            }
        )

    # Convert fold results into a DataFrame.
    fold_results = pd.DataFrame(results)

    # Store experiment configuration in the summary.
    metadata = {
        "model": "unmodified eLCS",
        "data": "data/raw/train.csv",
        "preprocessing": "training_only_exact_dedup" if source is not None else "none",
        "validation": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "learning_iterations": iterations,
        "population_size": POPULATION_SIZE,
    }

    # Calculate mean and standard deviation across the five folds.
    # These are fold statistics, not confidence intervals.
    summary = pd.DataFrame(
        [
            {
                **metadata,
                "fold": statistic,
                **fold_results[list(METRICS)]
                .agg(statistic)
                .to_dict(),
            }
            for statistic in ("mean", "std")
        ]
    )

    # Combine fold-level results with mean and standard deviation.
    output = pd.concat(
        [fold_results, summary],
        ignore_index=True,
    )

    # Create the reports directory if necessary.
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the results.
    output.to_csv(
        report_path,
        index=False,
        float_format="%.6f",
    )

    print(
        f"Saved {N_SPLITS}-fold {args.data} eLCS results "
        f"to {report_path}"
    )

    print(
        f"Balanced accuracy: "
        f"{fold_results['balanced_accuracy'].mean():.3f} "
        f"(+/- {fold_results['balanced_accuracy'].std():.3f})"
    )


if __name__ == "__main__":
    main()  