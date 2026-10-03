"""Development-only grouped CV, followed by an explicitly requested final test.

The reserved rows are not retroactively untouched by earlier full-data analyses.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__:
    from .model_data import DEFAULT_DATA_PATH, load_model_data, model_folds
    from . import elcs_runtime
else:
    from model_data import DEFAULT_DATA_PATH, load_model_data, model_folds
    import elcs_runtime

ELCS_MODELS = ("elcs", "elcs_dedup", "ensemble", "ensemble_dedup")
MODELS = (*ELCS_MODELS, "logistic_regression", "svm", "random_forest")
SPLIT = {"n_splits": 5, "random_state": 42, "outer_fold": 1}
REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "holdout"


def create_splits(X, y, groups):
    """Reserve the first outer fold; return inner folds in original row positions."""
    if set(np.unique(y)) != {0, 1} or len(np.unique(groups)) < 5:
        raise ValueError("Binary classes and at least five groups are required.")
    split_args = {key: SPLIT[key] for key in ("n_splits", "random_state")}
    dev, test = list(model_folds(X, y, groups, **split_args))[SPLIT["outer_fold"] - 1]
    if len(np.unique(groups.iloc[dev])) < 5:
        raise ValueError("Development data require at least five groups.")
    folds = [(dev[train], dev[valid]) for train, valid in
             model_folds(X.iloc[dev], y.iloc[dev], groups.iloc[dev], **split_args)]
    for train, valid in [(dev, test), *folds]:
        if any(set(np.unique(y.iloc[indices])) != {0, 1} for indices in (train, valid)):
            raise ValueError("Every training and validation partition must contain both classes.")
        if not set(groups.iloc[train]).isdisjoint(groups.iloc[valid]):
            raise ValueError("Groups overlap across a partition.")
    return dev, test, folds


def model_configuration(name, iterations, seeds, *, population_size=100, library_variant="corrected"):
    """Capture the model parameters before final-test access."""
    if name not in MODELS or iterations <= 0:
        raise ValueError("Unknown model or non-positive iteration budget.")
    if len(seeds) < 3 or len(seeds) % 2 != 1 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least three unique seeds, with an odd count.")
    if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size <= 0:
        raise ValueError("Population size must be a positive integer.")
    if library_variant not in elcs_runtime.VARIANTS:
        raise ValueError("Unknown eLCS library variant.")
    parameters = {
        "elcs": {"learning_iterations": iterations, "N": population_size, "random_state": 42},
        "elcs_dedup": {"learning_iterations": iterations, "N": population_size, "random_state": 42},
        "ensemble": {"learning_iterations": iterations, "N": population_size, "seeds": list(seeds)},
        "ensemble_dedup": {"learning_iterations": iterations, "N": population_size, "seeds": list(seeds)},
        "logistic_regression": {"max_iter": 1000, "random_state": 42},
        "svm": {"kernel": "rbf", "C": 1.0, "probability": True, "random_state": 42},
        "random_forest": {"n_estimators": 200, "n_jobs": -1, "random_state": 42},
    }
    config = {"name": name, "parameters": parameters[name]}
    if name in ELCS_MODELS:
        config.update(library_variant=library_variant, runtime=elcs_runtime.provenance(library_variant))
    return config


def elcs_model(parameters, library_variant="corrected", runtime=None):
    return elcs_runtime.create_model(parameters, library_variant, expected=runtime)


class Ensemble:
    """Fresh independently seeded eLCS members on each fit; binary majority vote."""
    def __init__(self, parameters, library_variant="corrected", runtime=None):
        self.parameters = parameters
        self.library_variant = library_variant
        self.runtime = runtime

    def fit(self, X, y):
        params = {k: v for k, v in self.parameters.items() if k != "seeds"}
        self.members = [elcs_model({**params, "random_state": seed},
                                   library_variant=self.library_variant, runtime=self.runtime).fit(X, y)
                        for seed in self.parameters["seeds"]]
        return self

    def predict(self, X):
        votes = np.vstack([member.predict(X) for member in self.members])
        return (votes.sum(axis=0) >= len(self.members) // 2 + 1).astype(int)


def make_model(config):
    name, params = config["name"], config["parameters"]
    if name in ELCS_MODELS:
        elcs_runtime.validate_configuration(config)
        if name in ("elcs", "elcs_dedup"):
            return elcs_model(params, library_variant=config["library_variant"], runtime=config["runtime"])
        return Ensemble(params, config["library_variant"], config["runtime"])
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(**params))
    if name == "random_forest":
        return RandomForestClassifier(**params)
    raise ValueError("Unknown frozen model.")


def split_table(length, dev, test, folds):
    table = pd.DataFrame({"row_id": np.arange(length), "partition": "development", "cv_fold": 0})
    table.loc[test, "partition"] = "test"
    for number, (_, valid) in enumerate(folds, 1):
        table.loc[valid, "cv_fold"] = number
    return table


def evaluate(config, X, y, source, train, valid, fold):
    original_count = len(train)
    if config["name"] in ("elcs_dedup", "ensemble_dedup"):
        train = train[~source.iloc[train].duplicated().to_numpy()]
    model = make_model(config)
    model.fit(X.iloc[train].to_numpy(), y.iloc[train].to_numpy())
    values = X.iloc[valid].to_numpy()
    predictions = model.predict(values)
    truth = y.iloc[valid].to_numpy()
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    result = {"fold": fold, "training_rows_raw": original_count, "training_rows": len(train),
              "evaluation_rows": len(valid), "true_negatives": tn, "false_positives": fp,
              "false_negatives": fn, "true_positives": tp,
              "accuracy": accuracy_score(truth, predictions),
              "balanced_accuracy": balanced_accuracy_score(truth, predictions),
              "precision": precision_score(truth, predictions, zero_division=0),
              "recall": recall_score(truth, predictions, zero_division=0),
              "f1": f1_score(truth, predictions, zero_division=0)}
    rows = pd.DataFrame({"row_id": valid, "fold": fold, "y_true": truth, "y_pred": predictions})
    # eLCS probabilities are deliberately omitted; do not mislabel vote fractions.
    if config["name"] not in ELCS_MODELS and hasattr(model, "predict_proba"):
        score = model.predict_proba(values)[:, 1]
        rows["score"] = score
        result.update(roc_auc=roc_auc_score(truth, score),
                      average_precision=average_precision_score(truth, score))
    return result, rows


def data_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_cv(model, output_dir=None, *, data_path=DEFAULT_DATA_PATH, iterations=1000, seeds=(11, 42, 73),
           population_size=100, library_variant="corrected"):
    """Freeze configuration and report CV only; never fit or predict reserved rows."""
    output = Path(output_dir) if output_dir is not None else REPORT_DIR / model
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    path = Path(data_path).resolve()
    config = {"model": model_configuration(model, iterations, seeds, population_size=population_size,
                                            library_variant=library_variant), "data_path": str(path),
              "data_sha256": data_hash(path), "sklearn_version": sklearn.__version__,
              "split": SPLIT.copy()}
    X, y, groups, source = load_model_data(path, include_source=True)
    dev, test, folds = create_splits(X, y, groups)
    if data_hash(path) != config["data_sha256"]:
        raise ValueError("Dataset changed during loading.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    split_table(len(X), dev, test, folds).to_csv(output / "splits.csv", index=False)
    results, predictions = [], []
    for number, (train, valid) in enumerate(folds, 1):
        result, rows = evaluate(config["model"], X, y, source, train, valid, number)
        results.append(result)
        predictions.append(rows)
    frame = pd.DataFrame(results)
    metric_names = (
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "roc_auc", "average_precision",
    )
    numeric = frame[[name for name in metric_names if name in frame]]
    summary = pd.DataFrame([{"fold": "mean", **numeric.mean().to_dict()},
                            {"fold": "std", **numeric.std(ddof=1).to_dict()}])
    pd.concat([frame, summary], ignore_index=True).to_csv(output / "cv_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "cv_predictions.csv", index=False)
    # Written last: incomplete CV or later edits cannot unlock final testing.
    completed = {
        name: data_hash(output / name)
        for name in ("config.json", "splits.csv", "cv_results.csv", "cv_predictions.csv")
    }
    (output / "cv_complete.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    print(f"Development CV saved to {output}; final test requires the test subcommand.")


def run_test(experiment):
    """Read frozen settings only; one final fit on development and one reserved test."""
    output = Path(experiment)
    artifacts = ("test_started.json", "test_results.csv", "test_predictions.csv")
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError("Final test was already started; existing results will not be overwritten.")
    completion = output / "cv_complete.json"
    if not completion.is_file():
        raise ValueError("Complete development CV before requesting final testing.")
    completed = json.loads(completion.read_text(encoding="utf-8"))
    for name in ("config.json", "splits.csv", "cv_results.csv", "cv_predictions.csv"):
        if not (output / name).is_file() or data_hash(output / name) != completed.get(name):
            raise ValueError("Frozen configuration or CV artifacts changed; use the original experiment.")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config["sklearn_version"] != sklearn.__version__ or config["split"] != SPLIT:
        raise ValueError("Split environment differs from the frozen configuration.")
    if config["model"]["name"] in ELCS_MODELS:
        elcs_runtime.validate_configuration(config["model"])
    path = Path(config["data_path"])
    if data_hash(path) != config["data_sha256"]:
        raise ValueError("Dataset differs from the frozen configuration.")
    if not all((output / name).is_file() for name in ("cv_results.csv", "cv_predictions.csv")):
        raise ValueError("Complete development CV before requesting final testing.")
    X, y, groups, source = load_model_data(path, include_source=True)
    dev, test, folds = create_splits(X, y, groups)
    if data_hash(path) != config["data_sha256"]:
        raise ValueError("Dataset changed during loading.")
    pd.testing.assert_frame_equal(pd.read_csv(output / "splits.csv"), split_table(len(X), dev, test, folds))
    # Exclusive marker also protects an interrupted final fit from accidental reruns.
    with (output / "test_started.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    result, rows = evaluate(config["model"], X, y, source, dev, test, "test")
    pd.DataFrame([result]).to_csv(output / "test_results.csv", index=False)
    rows.to_csv(output / "test_predictions.csv", index=False)
    print(f"Final test saved to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    cv = commands.add_parser("cv", help="Evaluate models only inside development rows")
    cv.add_argument("--model", choices=MODELS, required=True)
    cv.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    cv.add_argument("--iterations", type=int, default=1000)
    cv.add_argument("--population-size", type=int, default=100)
    cv.add_argument("--library-variant", choices=elcs_runtime.VARIANTS, default="corrected")
    cv.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 73])
    cv.add_argument("--output-dir", type=Path)
    final = commands.add_parser("test", help="Run a frozen experiment's final test once")
    final.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "cv":
        run_cv(args.model, args.output_dir, data_path=args.data, iterations=args.iterations, seeds=args.seeds,
               population_size=args.population_size, library_variant=args.library_variant)
    else:
        run_test(args.experiment)


if __name__ == "__main__":
    main()
