"""Evaluate the unmodified eLCS baseline on the raw or preprocessed dataset."""

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
from sklearn.model_selection import StratifiedGroupKFold

if __package__:
    from .model_data import load_model_data
else:
    from model_data import load_model_data


RANDOM_STATE = 42
LEARNING_ITERATIONS = 1000
POPULATION_SIZE = 100
N_SPLITS = 5

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"

PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "train_preprocessed.csv"
)

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
        help="Dataset to evaluate (default: %(default)s).",
    )

    args = parser.parse_args(argv)

    if args.iterations <= 0:
        parser.error("--iterations must be a positive integer")

    iterations = args.iterations

    # Select the dataset and report name.
    if args.data == "preprocessed":
        data_path = PROCESSED_DATA_PATH
        report_path = (
            REPORT_DIR
            / f"elcs_preprocessed_{iterations}_iterations.csv"
        )
    else:
        data_path = None
        report_path = (
            REPORT_DIR
            / f"elcs_raw_{iterations}_iterations.csv"
        )

    # Load the selected dataset.
    # Passing None uses the raw dataset through model_data.py.
    X, y, groups = load_model_data(data_path)

    X_values = X.to_numpy()
    y_values = y.to_numpy()
    group_values = groups.to_numpy()

    # Keep data, row order, grouping and split settings identical across models.
    # Feature groups do not guarantee separation of related material families.
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(
            X_values,
            y_values,
            groups=group_values,
        ),
        start=1,
    ):
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
                "data": (
                    str(data_path)
                    if data_path
                    else "data/raw/train.csv"
                ),
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
        "data": (
            str(data_path)
            if data_path
            else "data/raw/train.csv"
        ),
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