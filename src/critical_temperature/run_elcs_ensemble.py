"""Evaluate a multi-seed eLCS ensemble using majority voting."""

import argparse
from pathlib import Path

import numpy as np
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

DEFAULT_SEEDS = [11, 42, 73]

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)


def majority_vote(predictions: list[np.ndarray]) -> np.ndarray:
    """Combine predictions from an odd number of binary classifiers."""

    prediction_matrix = np.vstack(predictions)

    votes_for_one = prediction_matrix.sum(axis=0)

    required_votes = len(predictions) // 2 + 1

    return (votes_for_one >= required_votes).astype(int)


def main(argv: list[str] | None = None) -> None:
    """Evaluate an ensemble of independently seeded eLCS models."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--iterations",
        type=int,
        default=LEARNING_ITERATIONS,
        help="Learning iterations for each eLCS model.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds used for ensemble members.",
    )

    args = parser.parse_args(argv)

    if args.iterations <= 0:
        parser.error("--iterations must be a positive integer")

    if len(args.seeds) < 3:
        parser.error("At least three seeds are required")

    if len(args.seeds) % 2 == 0:
        parser.error("Use an odd number of seeds to avoid voting ties")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")

    X, y, groups = load_model_data()

    X_values = X.to_numpy()
    y_values = y.to_numpy()
    group_values = groups.to_numpy()

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

        print(f"\nFold {fold}/{N_SPLITS}")

        member_predictions = []
        member_balanced_accuracies = []

        for seed in args.seeds:

            print(
                f"  Training eLCS seed {seed} "
                f"for {args.iterations} iterations...",
                flush=True,
            )

            model = eLCS(
                learning_iterations=args.iterations,
                N=POPULATION_SIZE,
                random_state=seed,
            )

            model.fit(
                X_values[train_indices],
                y_values[train_indices],
            )

            predictions = model.predict(
                X_values[test_indices]
            )

            member_predictions.append(predictions)

            member_balanced_accuracies.append(
                balanced_accuracy_score(
                    y_values[test_indices],
                    predictions,
                )
            )

        ensemble_predictions = majority_vote(
            member_predictions
        )

        prediction_matrix = np.vstack(
            member_predictions
        )

        # A sample is counted as disagreement when the
        # ensemble members do not all make the same prediction.
        disagreement_rate = np.mean(
            np.ptp(prediction_matrix, axis=0) != 0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_values[test_indices],
            ensemble_predictions,
            labels=[0, 1],
        ).ravel()

        result = {
            "model": "eLCS multi-seed ensemble",
            "validation": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "fold": fold,
            "learning_iterations": args.iterations,
            "population_size": POPULATION_SIZE,
            "ensemble_size": len(args.seeds),
            "seeds": ",".join(
                str(seed) for seed in args.seeds
            ),
            "training_rows": len(train_indices),
            "test_rows": len(test_indices),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "accuracy": accuracy_score(
                y_values[test_indices],
                ensemble_predictions,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_values[test_indices],
                ensemble_predictions,
            ),
            "precision": precision_score(
                y_values[test_indices],
                ensemble_predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_values[test_indices],
                ensemble_predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_values[test_indices],
                ensemble_predictions,
                zero_division=0,
            ),
            "member_mean_balanced_accuracy": np.mean(
                member_balanced_accuracies
            ),
            "member_std_balanced_accuracy": np.std(
                member_balanced_accuracies,
                ddof=1,
            ),
            "disagreement_rate": disagreement_rate,
        }

        results.append(result)

        print(
            "  Ensemble balanced accuracy: "
            f"{result['balanced_accuracy']:.3f}"
        )

        print(
            "  Member disagreement rate: "
            f"{disagreement_rate:.3f}"
        )

    fold_results = pd.DataFrame(results)

    summary_columns = [
        *METRICS,
        "member_mean_balanced_accuracy",
        "member_std_balanced_accuracy",
        "disagreement_rate",
    ]

    metadata = {
        "model": "eLCS multi-seed ensemble",
        "validation": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "learning_iterations": args.iterations,
        "population_size": POPULATION_SIZE,
        "ensemble_size": len(args.seeds),
        "seeds": ",".join(
            str(seed) for seed in args.seeds
        ),
    }

    summary = pd.DataFrame(
        [
            {
                **metadata,
                "fold": statistic,
                **fold_results[
                    summary_columns
                ].agg(statistic).to_dict(),
            }
            for statistic in ("mean", "std")
        ]
    )

    output = pd.concat(
        [fold_results, summary],
        ignore_index=True,
    )

    report_path = (
        REPORT_DIR
        / f"elcs_ensemble_{args.iterations}_iterations.csv"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        report_path,
        index=False,
        float_format="%.6f",
    )

    print(
        f"\nSaved ensemble results to {report_path}"
    )

    print(
        "Mean ensemble balanced accuracy: "
        f"{fold_results['balanced_accuracy'].mean():.3f}"
    )

    print(
        "Mean disagreement rate: "
        f"{fold_results['disagreement_rate'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
    