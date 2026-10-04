"""Authorized exploratory retraining; reused-test scores are descriptive, not confirmatory."""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

if __package__:
    from .model_data import DEFAULT_DATA_PATH, load_model_data
    from .run_holdout import create_splits, data_hash, run_cv, run_test
    from .run_development_experiments import SEEDS, SELECTION_RULE, read_summary, source_fingerprint, write_json
    from .compare_models import comparison_table
else:
    from model_data import DEFAULT_DATA_PATH, load_model_data
    from run_holdout import create_splits, data_hash, run_cv, run_test
    from run_development_experiments import SEEDS, SELECTION_RULE, read_summary, source_fingerprint, write_json
    from compare_models import comparison_table

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = tuple(ROOT / "reports/holdout/conventional_defaults" / name
                   for name in ("logistic_regression", "svm", "random_forest"))
PASSES = (1, 5, 20)
POPULATIONS = (100, 1000)


def artifact_hashes(directory):
    """Bind all files, including completion manifests and exported rule evidence."""
    return {path.relative_to(directory).as_posix(): data_hash(path)
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def reference_hashes():
    for directory in REFERENCES:
        if not directory.is_dir() or not any(directory.iterdir()):
            raise ValueError(f"Missing historical reference: {directory}")
    return {
        "experiments": {str(path.resolve()): artifact_hashes(path) for path in REFERENCES},
    }


def training_counts(data_path):
    """Use the runner's exact source-row deduplication, never transformed features."""
    X, y, groups, source = load_model_data(data_path, include_source=True)
    dev, test, folds = create_splits(X, y, groups)
    dev_ids, test_ids = set(dev), set(test)
    validation = [int(row) for _, valid in folds for row in valid]
    if (dev_ids & test_ids or dev_ids | test_ids != set(range(len(X)))
            or len(dev_ids) != len(dev) or len(test_ids) != len(test)
            or len(validation) != len(dev) or set(validation) != dev_ids):
        raise ValueError("Split coverage must include every row exactly once.")
    for train, valid in folds:
        if (set(train) & set(valid) or set(train) | set(valid) != dev_ids
                or len(set(train)) != len(train) or len(set(valid)) != len(valid)):
            raise ValueError("CV training/validation coverage differs from development rows.")

    def count(indices):
        return int((~source.iloc[indices].duplicated()).sum())

    final = count(dev)
    cv = [count(train) for train, _ in folds]
    if not final or not cv or not all(cv):
        raise ValueError("Nonempty deduplicated training partitions are required.")
    return {"final_raw": len(dev), "final_deduplicated": final,
            "cv_raw": [len(train) for train, _ in folds], "cv_deduplicated": cv,
            "development_validation_rows": len(validation), "reused_test_rows": len(test)}


def select_candidate(rows):
    """Exact mean-CV ties only; never use test scores for selection."""
    if not rows or not all(np.isfinite(row["balanced_accuracy"]) for row in rows):
        raise ValueError("Selection requires finite development balanced accuracies.")
    return min(rows, key=lambda row: (-row["balanced_accuracy"],
               row["iterations_per_member"], row["population_size_per_member"]))


def run_study(output_dir, *, data_path=DEFAULT_DATA_PATH):
    """No retries/resumption: any interrupted run requires a new output directory."""
    output, data = Path(output_dir).resolve(), Path(data_path).resolve()
    for original in REFERENCES:
        original = original.resolve()
        if output == original or original in output.parents:
            raise ValueError("Output must not be inside an original reference.")
    output.mkdir(parents=True, exist_ok=False)
    completed, results = [], []
    active = "planning"

    def status(state, **extra):
        write_json(output / "status.json", {"state": state, "active": active,
                   "completed": completed.copy(), **extra})

    try:
        status("active")
        sources, dataset, originals = source_fingerprint(), data_hash(data), reference_hashes()

        def check():
            if source_fingerprint() != sources:
                raise RuntimeError("Source changed during exploratory revision.")
            if data_hash(data) != dataset:
                raise RuntimeError("Dataset changed during exploratory revision.")
            if reference_hashes() != originals:
                raise RuntimeError("Historical reference artifacts changed during exploratory revision.")

        check()
        counts = training_counts(data)
        check()
        budgets = [{"final_development_passes": passes, "iterations_per_member":
                    passes * counts["final_deduplicated"], "population_size_per_member": population,
                    "cv_pass_equivalents": [passes * counts["final_deduplicated"] / count
                                            for count in counts["cv_deduplicated"]]}
                   for passes in PASSES for population in POPULATIONS]
        plan = {
            "exploratory": True,
            "authorization": "User explicitly authorized same-dataset exploratory revision, six CV candidates and two once-only reused-test scores.",
            "test_exposure": "The reserved test is reused and has already been exposed; earlier full-data analyses also prevent an untouched-test claim.",
            "interpretation": "Descriptive exploratory scores only; no new inferential test. The original statistical_test endpoint and recipe remain unchanged.",
            "selection_rule": SELECTION_RULE, "single_seed": 42, "ensemble_seeds": list(SEEDS),
            "data_path": str(data), "data_sha256": dataset, "source_sha256": sources,
            "training_counts": counts, "budgets": budgets,
            "budget_rule": "Constant integer iterations across CV folds and final fit: passes times exact final-development source-row dedup count.",
            "ensemble_budget": "Three independently seeded members; three times the single model's total iterations.",
            "historical_references_sha256": originals,
            "reporting_rule": "Five-row validated descriptive CV/test tables: revised single/ensemble then three fixed conventional references; no test-driven reselection. Old eLCS findings are retained in parent-owned prose, not bound as artifacts.",
            "restart_policy": "Fresh directory only; no automatic retry/resume or removal of interrupted-run markers.",
        }
        write_json(output / "plan.json", plan)
        plan_digest = data_hash(output / "plan.json")

        def stage(name, action, evidence):
            nonlocal active
            active = name
            check()
            if data_hash(output / "plan.json") != plan_digest:
                raise RuntimeError("Frozen exploratory plan changed.")
            status("active")
            started = time.perf_counter()
            print(f"START exploratory {name}", flush=True)
            action()
            check()
            if data_hash(output / "plan.json") != plan_digest:
                raise RuntimeError("Frozen exploratory plan changed.")
            completed.append({"stage": name, "elapsed_wall_seconds": time.perf_counter() - started,
                              "artifacts_sha256": evidence()})
            status("active")
            print(f"DONE exploratory {name}", flush=True)

        def cv(name, model, iterations, population):
            experiment = output / name
            stage(name, lambda: run_cv(model, experiment, data_path=data, iterations=iterations,
                  seeds=SEEDS, population_size=population, library_variant="corrected"),
                  lambda: artifact_hashes(experiment))
            results.append(read_summary(experiment, name, model, "corrected", iterations, population))
            pd.DataFrame(results).to_csv(output / "development_summary.csv", index=False)

        for budget in budgets:
            iterations, population = budget["iterations_per_member"], budget["population_size_per_member"]
            cv(f"corrected_dedup_i{iterations}_n{population}", "elcs_dedup", iterations, population)
        best = select_candidate(results)
        write_json(output / "selection.json", {"selection_rule": SELECTION_RULE,
                   "selected_single_model": best, "plan_sha256": plan_digest,
                   "candidate_cv_artifacts_sha256": {row["experiment"]: artifact_hashes(output / row["experiment"])
                                                     for row in results}})
        ensemble_name = "corrected_ensemble_dedup_selected"
        cv(ensemble_name, "ensemble_dedup", best["iterations_per_member"], best["population_size_per_member"])
        selected = [output / best["experiment"], output / ensemble_name]
        experiments = [*selected, *REFERENCES]

        def table(kind):
            destination = output / f"descriptive_{kind}_table.csv"
            frame = comparison_table(experiments, stage=kind, output=destination)
            frame["evidence_role"] = ["exploratory_revision"] * 2 + ["fixed_historical_reference"] * 3
            frame["interpretation"] = "descriptive_only_no_new_inference"
            frame["reused_test_exploratory"] = kind == "test"
            frame.to_csv(destination, index=False)

        stage("descriptive_cv_table", lambda: table("cv"),
              lambda: {"descriptive_cv_table.csv": data_hash(output / "descriptive_cv_table.csv")})
        reporting = {
            "reused_test_exploratory": True, "no_new_inferential_test": True,
            "selection_rule": SELECTION_RULE, "plan_sha256": plan_digest,
            "selection_sha256": data_hash(output / "selection.json"),
            "test_order": [path.name for path in selected],
            "selected_cv_artifacts_sha256": {path.name: artifact_hashes(path) for path in selected},
            "historical_references_sha256": originals,
            "reporting_rule": plan["reporting_rule"], "no_test_driven_reselection": True,
        }
        write_json(output / "reporting_record.json", reporting)
        reporting_digest = data_hash(output / "reporting_record.json")
        for experiment in selected:
            def score(experiment=experiment):
                if (data_hash(output / "reporting_record.json") != reporting_digest
                        or data_hash(output / "selection.json") != reporting["selection_sha256"]):
                    raise RuntimeError("Pre-test selection/reporting record changed.")
                for name, artifacts in reporting["selected_cv_artifacts_sha256"].items():
                    for relative, digest in artifacts.items():
                        if data_hash(output / name / relative) != digest:
                            raise RuntimeError("Selected CV/config evidence changed before exploratory scoring.")
                run_test(experiment)
            stage(f"reused_test_exploratory_{experiment.name}", score,
                  lambda experiment=experiment: artifact_hashes(experiment))
        stage("descriptive_test_table", lambda: table("test"),
              lambda: {"descriptive_test_table.csv": data_hash(output / "descriptive_test_table.csv")})
        check()
        active = None
        status("completed")
        write_json(output / "study_complete.json", {
            "exploratory": True, "reused_test_exploratory": True, "no_new_inferential_test": True,
            "completed": completed, "selected_single_experiment": best["experiment"],
            "ensemble_experiment": ensemble_name, "artifacts_sha256": artifact_hashes(output),
        })
    except BaseException as error:
        status("failed_or_interrupted", error=repr(error))
        raise
    return pd.DataFrame(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)
    run_study(args.output_dir, data_path=args.data)


if __name__ == "__main__":
    main()
