"""Run a bounded eLCS development study without evaluating the reserved holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

if __package__:
    from .model_data import DEFAULT_DATA_PATH
    from .run_holdout import run_cv
else:
    from model_data import DEFAULT_DATA_PATH
    from run_holdout import run_cv


BUDGETS = ((1000, 100), (1000, 1000), (10000, 100), (10000, 1000))
SEEDS = (11, 42, 73)
SELECTION_RULE = "Highest mean development balanced accuracy; ties prefer fewer iterations, then smaller population."


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/critical_temperature").glob("*.py"))
    paths += sorted((root / "third_party/scikit-eLCS/skeLCS").glob("*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def read_summary(experiment, name, model, variant, iterations, population_size):
    frame = pd.read_csv(experiment / "cv_results.csv", dtype={"fold": str}).set_index("fold")
    metrics = ("accuracy", "balanced_accuracy", "precision", "recall", "f1")
    return {"experiment": name, "model": model, "library_variant": variant,
            "iterations_per_member": iterations, "population_size_per_member": population_size,
            "members": len(SEEDS) if model == "ensemble_dedup" else 1,
            **frame.loc["mean", list(metrics)].astype(float).to_dict(),
            "balanced_accuracy_std": float(frame.loc["std", "balanced_accuracy"])}


def run_study(output_dir, *, data_path=DEFAULT_DATA_PATH):
    """Save the declared plan first, run CV, then select the ensemble budget."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_fingerprint()
    plan = {
        "stage": "development_cv_only", "data_path": str(Path(data_path).resolve()),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "source_sha256": sources, "budgets": [list(budget) for budget in BUDGETS],
        "selection_rule": SELECTION_RULE, "ensemble_seeds": list(SEEDS),
        "warning": "The ensemble uses three times the total iteration budget of a single model.",
    }
    write_json(output / "plan.json", plan)
    results = []
    completed = []

    def experiment(name, model, variant, iterations, population_size):
        if source_fingerprint() != sources:
            raise RuntimeError("Source changed during the study; do not mix implementations.")
        if hashlib.sha256(Path(data_path).read_bytes()).hexdigest() != plan["data_sha256"]:
            raise RuntimeError("Dataset changed during the study.")
        write_json(output / "status.json", {"state": "running", "active": name,
                                           "completed": completed.copy()})
        print(f"START {name}", flush=True)
        run_cv(model, output / name, data_path=data_path, iterations=iterations,
               seeds=SEEDS, population_size=population_size, library_variant=variant)
        if source_fingerprint() != sources:
            raise RuntimeError("Source changed while the experiment was running.")
        results.append(read_summary(output / name, name, model, variant, iterations, population_size))
        completed.append(name)
        pd.DataFrame(results).to_csv(output / "development_summary.csv", index=False)
        print(f"DONE {name}: balanced accuracy {results[-1]['balanced_accuracy']:.6f}", flush=True)

    try:
        experiment("unmodified_raw_i1000_n100", "elcs", "unmodified", 1000, 100)
        experiment("unmodified_dedup_i1000_n100", "elcs_dedup", "unmodified", 1000, 100)
        experiment("corrected_raw_i1000_n100", "elcs", "corrected", 1000, 100)
        for iterations, population in BUDGETS:
            experiment(f"corrected_dedup_i{iterations}_n{population}", "elcs_dedup",
                       "corrected", iterations, population)
        candidates = [row for row in results if row["model"] == "elcs_dedup"
                      and row["library_variant"] == "corrected"]
        best = min(candidates, key=lambda row: (-row["balanced_accuracy"],
                   row["iterations_per_member"], row["population_size_per_member"]))
        write_json(output / "selection.json", {
            "selection_rule": SELECTION_RULE, "selected_single_model": best,
        })
        experiment("corrected_ensemble_dedup_selected", "ensemble_dedup", "corrected",
                   best["iterations_per_member"], best["population_size_per_member"])
        ensemble = results[-1]
        write_json(output / "study_complete.json", {
            "completed": completed, "selected_single_experiment": best["experiment"],
            "ensemble_experiment": ensemble["experiment"],
            "development_balanced_accuracy_difference": ensemble["balanced_accuracy"] - best["balanced_accuracy"],
            "reserved_holdout_evaluated": False,
        })
        write_json(output / "status.json", {"state": "completed", "completed": completed})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed_or_interrupted",
                                           "completed": completed, "error": repr(error)})
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
