"""Build compatible experiment tables and test one predeclared final comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_integer_dtype
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

if __package__:
    from . import elcs_runtime
    from .model_data import load_model_data
    from .run_holdout import SPLIT, create_splits, data_hash, split_table
else:
    import elcs_runtime
    from model_data import load_model_data
    from run_holdout import SPLIT, create_splits, data_hash, split_table

METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")


def load_experiment(directory, stage):
    """Validate saved artifacts and attach source feature groups to predictions."""
    if stage not in ("cv", "test"):
        raise ValueError("Stage must be cv or test.")
    directory = Path(directory)
    completion_path = directory / "cv_complete.json"
    if not completion_path.is_file():
        raise ValueError(f"Complete development CV is required: {directory}")
    completed = json.loads(completion_path.read_text(encoding="utf-8"))
    for name in ("config.json", "splits.csv", "cv_results.csv", "cv_predictions.csv"):
        path = directory / name
        if not path.is_file() or data_hash(path) != completed.get(name):
            raise ValueError(f"Changed or incomplete CV artifact: {path}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = config["model"]
    if model["name"] in ("elcs", "elcs_dedup", "ensemble", "ensemble_dedup"):
        elcs_runtime.validate_configuration(model)
    elif model["name"] not in ("logistic_regression", "svm", "random_forest"):
        raise ValueError("Unknown recorded model.")
    source = Path(config["data_path"])
    if data_hash(source) != config["data_sha256"]:
        raise ValueError("Source dataset changed since the experiment.")
    if config["sklearn_version"] != sklearn.__version__ or config["split"] != SPLIT:
        raise ValueError("Use the experiment's recorded split environment and recipe.")
    if stage == "test":
        paths = ("test_started.json", "test_results.csv", "test_predictions.csv", "test_complete.json")
        if not all((directory / name).is_file() for name in paths):
            raise ValueError(f"Complete final evaluation is required: {directory}")
        started = json.loads((directory / "test_started.json").read_text(encoding="utf-8"))
        if started != config:
            raise ValueError("Final evaluation used a different configuration.")
    X, y, groups = load_model_data(source)
    if data_hash(source) != config["data_sha256"]:
        raise ValueError("Source dataset changed during loading.")
    dev, test, folds = create_splits(X, y, groups)
    assignments = pd.read_csv(directory / "splits.csv")
    try:
        pd.testing.assert_frame_equal(assignments, split_table(len(X), dev, test, folds))
    except AssertionError as error:
        raise ValueError("Saved row assignments differ from the current split recipe.") from error
    # Preserve adjacent float scores: changing ties changes ROC-AUC and average precision.
    predictions = pd.read_csv(directory / f"{stage}_predictions.csv", float_precision="round_trip")
    required = {"row_id", "fold", "y_true", "y_pred"}
    if not required.issubset(predictions) or predictions.empty:
        raise ValueError("Prediction records are missing required columns or rows.")
    ids = predictions.row_id
    expected = dev if stage == "cv" else test
    if not is_integer_dtype(ids) or ids.duplicated().any() or set(ids) != set(expected):
        raise ValueError("Prediction row IDs must cover the evaluation partition exactly once.")
    if not predictions[["y_true", "y_pred"]].isin([0, 1]).all().all():
        raise ValueError("Predictions and labels must be binary and non-missing.")
    truth = y.iloc[ids.to_numpy()].to_numpy()
    if not np.array_equal(predictions.y_true.to_numpy(), truth):
        raise ValueError("Prediction labels disagree with the original dataset.")
    expected_fold = assignments.set_index("row_id").loc[ids, "cv_fold"].to_numpy()
    if stage == "cv" and not np.array_equal(predictions.fold.to_numpy(), expected_fold):
        raise ValueError("Prediction folds disagree with saved validation assignments.")
    if stage == "test" and not predictions.fold.eq("test").all():
        raise ValueError("Final prediction records must be marked as test, not CV.")
    if "score" in predictions:
        if model["name"] in ("elcs", "elcs_dedup", "ensemble", "ensemble_dedup"):
            raise ValueError("eLCS records must not include probability scores.")
        scores = predictions.score.to_numpy()
        if not np.isfinite(scores).all() or not predictions.score.between(0, 1).all():
            raise ValueError("Probability scores must be finite and between zero and one.")
    if stage == "cv":
        validate_cv_results(directory, predictions)
    if stage == "test":
        try:
            results = pd.read_csv(directory / "test_results.csv")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            raise ValueError("Final results are empty or malformed.") from error
        if len(results) != 1 or "fold" not in results or results.iloc[0]["fold"] != "test":
            raise ValueError("Final results must contain exactly one test row.")
        tn, fp, fn, tp = confusion_matrix(truth, predictions.y_pred, labels=[0, 1]).ravel()
        expected_results = {
            "evaluation_rows": len(predictions), "true_negatives": tn,
            "false_positives": fp, "false_negatives": fn, "true_positives": tp,
            **metrics(predictions),
        }
        for name, value in expected_results.items():
            try:
                matches = name in results and np.isclose(
                    float(results.iloc[0][name]), value, rtol=1e-12, atol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError("Final results disagree with the saved predictions.")
    if stage == "test":
        manifest = json.loads((directory / "test_complete.json").read_text(encoding="utf-8"))
        required = {"config.json", "test_started.json", "test_results.csv", "test_predictions.csv"}
        if not required.issubset(manifest):
            raise ValueError("Final completion manifest is incomplete.")
        for name, digest in manifest.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid final artifact path.")
            path = directory / relative
            if not path.is_file() or data_hash(path) != digest:
                raise ValueError(f"Final artifact changed or missing: {name}")
        if model["name"] in ("elcs", "elcs_dedup", "ensemble", "ensemble_dedup"):
            if __package__:
                from .extract_elcs_rules import inspect_experiment
            else:
                from extract_elcs_rules import inspect_experiment
            inspect_experiment(directory, quiet=True)
    predictions = predictions.assign(group_id=groups.iloc[ids.to_numpy()].to_numpy())
    return config, predictions.sort_values("row_id").reset_index(drop=True)


def validate_cv_results(directory, predictions):
    """Cross-check every recorded fold and its metric summaries."""
    try:
        results = pd.read_csv(directory / "cv_results.csv")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError("CV results are empty or malformed.") from error
    expected = []
    for fold, rows in predictions.groupby("fold"):
        tn, fp, fn, tp = confusion_matrix(rows.y_true, rows.y_pred, labels=[0, 1]).ravel()
        expected.append({"fold": str(fold), "evaluation_rows": len(rows),
                         "true_negatives": tn, "false_positives": fp,
                         "false_negatives": fn, "true_positives": tp, **metrics(rows)})
    frame = pd.DataFrame(expected)
    numeric = frame[[name for name in METRICS if name in frame]]
    expected.extend([{"fold": "mean", **numeric.mean().to_dict()},
                     {"fold": "std", **numeric.std(ddof=1).to_dict()}])
    if "fold" not in results or results.fold.astype(str).tolist() != [row["fold"] for row in expected]:
        raise ValueError("CV results must cover each fold and mean/std exactly once.")
    for index, record in enumerate(expected):
        for name, value in record.items():
            if name == "fold":
                continue
            try:
                matches = name in results and np.isclose(float(results.iloc[index][name]), value,
                                                        rtol=1e-12, atol=1e-12)
            except (ValueError, TypeError):
                matches = False
            if not matches:
                raise ValueError("CV results disagree with the saved predictions.")


def compatible(configs, predictions):
    """Require one dataset, split environment, evaluation population and labels."""
    first = configs[0]
    for config, rows in zip(configs[1:], predictions[1:]):
        if any(config[key] != first[key] for key in ("data_sha256", "sklearn_version", "split")):
            raise ValueError("Experiments must use compatible dataset and split settings.")
        columns = ["row_id", "fold", "y_true", "group_id"]
        if not rows[columns].equals(predictions[0][columns]):
            raise ValueError("Experiments do not evaluate the same rows, folds and labels.")


def metrics(rows):
    truth, predicted = rows.y_true, rows.y_pred
    result = {
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
        "f1": f1_score(truth, predicted, zero_division=0),
    }
    if "score" in rows:
        result.update(roc_auc=roc_auc_score(truth, rows.score),
                      average_precision=average_precision_score(truth, rows.score))
    return result


def comparison_table(experiments, *, stage, output):
    """CV metrics are unweighted fold means/SDs; final metrics use the holdout."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Comparison output already exists: {output}")
    if not experiments:
        raise ValueError("Provide at least one experiment.")
    loaded = [load_experiment(directory, stage) for directory in experiments]
    configs, predictions = zip(*loaded)
    compatible(configs, predictions)
    records = []
    for directory, config, rows in zip(experiments, configs, predictions):
        name = config["model"]["name"]
        preprocessing = "none"
        if name in ("elcs_dedup", "ensemble_dedup"):
            preprocessing = "training_only_exact_dedup"
        elif name in ("logistic_regression", "svm"):
            preprocessing = "training_only_standard_scaler"
        result = {
            "display_name": display_name(config["model"]),
            "experiment": str(directory), "model": name, "stage": stage,
            "source_variant": config["model"].get("library_variant", "sklearn"),
            "preprocessing": preprocessing,
            "parameters": json.dumps(config["model"]["parameters"], sort_keys=True),
            "dataset_sha256": config["data_sha256"],
            "evaluation_rows": len(rows), "evaluation_groups": rows.group_id.nunique(),
            "aggregation": "unweighted_cv_mean" if stage == "cv" else "heldout",
        }
        if stage == "cv":
            frame = pd.DataFrame([metrics(part) for _, part in rows.groupby("fold")])
            result.update(frame.mean().to_dict())
            result.update({f"{name}_std": value for name, value in frame.std(ddof=1).items()})
        else:
            result.update(metrics(rows))
        for name in METRICS:
            result.setdefault(name, np.nan)
            result.setdefault(f"{name}_std", np.nan)
        tn, fp, fn, tp = confusion_matrix(rows.y_true, rows.y_pred, labels=[0, 1]).ravel()
        result.update(true_negatives=int(tn), false_positives=int(fp),
                      false_negatives=int(fn), true_positives=int(tp))
        records.append(result)
    table = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, mode="x")
    return table


def statistical_test(baseline, improved, *, output, permutations=19999, random_state=42):
    """Test the predeclared deduplicated eLCS/ensemble contrast on final rows only."""
    if __package__:
        from .statistical_comparison import paired_group_permutation
    else:
        from statistical_comparison import paired_group_permutation
    if permutations != 19999 or random_state != 42:
        raise ValueError("The approved primary statistical recipe uses 19999 draws and seed 42.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Statistical output already exists: {output}")
    base_config, base = load_experiment(baseline, "test")
    improved_config, other = load_experiment(improved, "test")
    compatible([base_config, improved_config], [base, other])
    if base_config["model"]["name"] != "elcs_dedup" or improved_config["model"]["name"] != "ensemble_dedup":
        raise ValueError("The primary comparison is elcs_dedup versus ensemble_dedup.")
    if any(config["model"].get("library_variant") != "corrected"
           for config in (base_config, improved_config)):
        raise ValueError("The primary comparison requires corrected eLCS in both systems.")
    expected_single = {"learning_iterations": 1000, "N": 100, "random_state": 42}
    expected_ensemble = {"learning_iterations": 1000, "N": 100, "seeds": [11, 42, 73]}
    if (base_config["model"]["parameters"] != expected_single
            or improved_config["model"]["parameters"] != expected_ensemble):
        raise ValueError("The primary comparison must use the declared selected settings.")
    result = paired_group_permutation(
        base.y_true.to_numpy(), base.y_pred.to_numpy(), other.y_pred.to_numpy(),
        base.group_id.to_numpy(), permutations=permutations, random_state=random_state,
    )
    result.update(baseline_experiment=str(baseline), improved_experiment=str(improved),
                  dataset_sha256=base_config["data_sha256"], stage="test")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def display_name(model):
    names = {"elcs": "eLCS (raw)", "elcs_dedup": "eLCS (deduplicated)",
             "ensemble": "eLCS ensemble (raw)",
             "ensemble_dedup": "eLCS ensemble (deduplicated)",
             "logistic_regression": "Logistic regression", "svm": "SVM",
             "random_forest": "Random forest"}
    prefix = model.get("library_variant", "").capitalize()
    return f"{prefix} {names[model['name']]}".strip()


def markdown_summary(table):
    """Format headline metrics without an optional Markdown dependency."""
    columns = ["display_name", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in table.iterrows():
        values = [str(row.display_name).replace("|", "\\|"),
                  *["" if pd.isna(row[name]) else f"{row[name]:.4f}" for name in columns[1:]]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    table = commands.add_parser("table", help="Combine comparable development or final metrics")
    table.add_argument("--stage", choices=("cv", "test"), required=True)
    table.add_argument("--experiments", type=Path, nargs="+", required=True)
    table.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("stats", help="Group-paired permutation test on final predictions only")
    test.add_argument("--baseline", type=Path, required=True)
    test.add_argument("--improved", type=Path, required=True)
    test.add_argument("--permutations", type=int, default=19999)
    test.add_argument("--random-state", type=int, default=42)
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "table":
        comparison_table(args.experiments, stage=args.stage, output=args.output)
    else:
        statistical_test(args.baseline, args.improved, output=args.output,
                         permutations=args.permutations, random_state=args.random_state)
    print(f"Saved {args.command} output to {args.output}")


if __name__ == "__main__":
    main()
