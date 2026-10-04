"""Build an offline exploratory results page; never fit or score models."""

import ast
import csv
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[2]
FINAL = Path("reports/holdout/elcs_exploratory_revision_v2")
TEMPLATE = Path(__file__).with_name("results_page.html")


def read_json(path):
    if not path.is_file():
        raise ValueError(f"Evidence missing; complete the revision first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(root, name):
    path = (root / name.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe evidence path: {name}")
    return path


def recorded_path(root, name):
    """Resolve recorded machine-local report paths inside this checkout."""
    name = name.replace("\\", "/")
    if "/reports/" in name:
        name = "reports/" + name.split("/reports/", 1)[1]
    if not name.startswith("reports/"):
        raise ValueError(f"Not a repository report path: {name}")
    return safe_path(root, name)


def check_hash(path, digest):
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Evidence missing or changed (hash mismatch): {path}")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def percent(value):
    return f"{float(value) * 100:.1f}%"


def link(path, label):
    return f'<a href="{escape(str(path).replace(chr(92), "/"), quote=True)}">{escape(label)}</a>'


def build_page(root=ROOT):
    """Validate completed revision/reference hashes; return self-contained HTML."""
    root = Path(root).resolve()
    folder = root / FINAL
    complete = read_json(folder / "study_complete.json")
    for name, digest in complete["artifacts_sha256"].items():
        check_hash(safe_path(folder, name), digest)
    plan = read_json(folder / "plan.json")
    reporting = read_json(folder / "reporting_record.json")
    selection = read_json(folder / "selection.json")
    check_hash(folder / "plan.json", reporting["plan_sha256"])
    check_hash(folder / "selection.json", reporting["selection_sha256"])
    if (read_json(folder / "status.json")["state"] != "completed"
            or not complete["exploratory"] or not complete["no_new_inferential_test"]
            or not reporting["reused_test_exploratory"] or not plan["exploratory"]):
        raise ValueError("A completed, explicitly exploratory revision is required.")
    for name, artifacts in reporting["historical_references_sha256"]["experiments"].items():
        directory = recorded_path(root, name)
        for relative, digest in artifacts.items():
            check_hash(safe_path(directory, relative), digest)

    rows = read_csv(folder / "descriptive_test_table.csv")
    baseline_dir = safe_path(folder, complete["selected_single_experiment"])
    ensemble_dir = safe_path(folder, complete["ensemble_experiment"])
    baseline = next(row for row in rows if recorded_path(root, row["experiment"]) == baseline_dir)
    ensemble = next(row for row in rows if recorded_path(root, row["experiment"]) == ensemble_dir)
    best = max(rows, key=lambda row: float(row["balanced_accuracy"]))
    difference = float(ensemble["balanced_accuracy"]) - float(baseline["balanced_accuracy"])
    if len(rows) != 5 or best["model"] != "random_forest" or difference <= 0:
        raise ValueError("This presentation describes the completed five-model exploratory revision.")
    for row in rows:
        directory = recorded_path(root, row["experiment"])
        for manifest in ("cv_complete.json", "test_complete.json"):
            for name, digest in read_json(directory / manifest).items():
                check_hash(safe_path(directory, name), digest)

    training = read_csv(baseline_dir / "test_results.csv")[0]
    chosen = selection["selected_single_model"]
    cv_rows = read_csv(folder / "descriptive_cv_table.csv")
    cv_ensemble = next(row for row in cv_rows if recorded_path(root, row["experiment"]) == ensemble_dir)
    iterations, population = chosen["iterations_per_member"], chosen["population_size_per_member"]
    training_rows = int(training["training_rows"])
    passes = iterations / training_rows
    cv_training_rows = [int(float(row["training_rows"])) for row in read_csv(baseline_dir / "cv_results.csv")
                        if row["fold"] not in ("mean", "std")]
    negatives = int(baseline["true_negatives"]) + int(baseline["false_positives"])
    majority_accuracy = negatives / int(baseline["evaluation_rows"])

    bars, table, predictions = [], [], []
    for row in sorted(rows, key=lambda row: float(row["balanced_accuracy"]), reverse=True):
        score = float(row["balanced_accuracy"]) * 100
        style = "ensemble" if row is ensemble else "single" if row is baseline else ""
        name = escape(row["display_name"])
        bars.append(f'<li class="chart-row {style}" data-score="{score!r}"><div class="bar-label"><span>{name}</span><strong>{score:.1f}%</strong></div><div class="track" aria-hidden="true"><div class="bar" style="width:{score!r}%"></div></div></li>')
        table.append(f'<tr><th scope="row">{name}</th>' + ''.join(
            f'<td data-label="{label}">{percent(row[field])}</td>'
            for field, label in (("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))) + '</tr>')
        path = recorded_path(root, row["experiment"]).relative_to(root) / "test_predictions.csv"
        predictions.append('<li>' + link(path, row["display_name"]) + '</li>')

    rules = []
    for seed, phenotype in ((11, 1), (42, 0), (73, 1)):
        population_path = ensemble_dir.relative_to(root) / f"rules/member_{seed}/rules.csv"
        candidates = [(number, row) for number, row in enumerate(read_csv(root / population_path), 2)
                      if int(float(row["above_77k"])) == phenotype]
        csv_row, row = max(candidates, key=lambda pair: (int(pair[1]["Match Count"]),
                                                       float(pair[1]["Fitness"]), -pair[0]))
        names = row["Specified Attribute Names"].split(", ")
        values = ast.literal_eval("[" + row["Specified Values"] + "]")
        conditions = dict(zip(names, values, strict=True))
        excerpts = []
        for field, value in list(conditions.items())[:3]:
            condition = f"between {value[0]:.4f} and {value[1]:.4f} (strict)" if isinstance(value, list) else f"equal to {value:g}"
            excerpts.append(f'<li><code>{escape(field)}</code>: {condition}</li>')
        predicted = "above 77 K" if phenotype else "at or below 77 K"
        rules.append(f'''<details class="rule-example"><summary>Seed {seed}: predicts {predicted}<span>{int(row["Correct Count"])} correct / {int(row["Match Count"])} training matches</span></summary>
<div class="disclosure"><p>This rule specifies <strong>{len(names)} of 81 features</strong>. Three condition excerpts:</p><ul class="conditions">{''.join(excerpts)}</ul>
<p>All specified conditions must match together, including specified conditions omitted from these rounded excerpts. Only unspecified features are unrestricted.</p>
<p>Training accuracy: <strong>{percent(row["Accuracy"])}</strong>. Fitness: {float(row["Fitness"]):.5f}. Numerosity: {int(row["Numerosity"])}. Repeated presentations can contribute multiple training matches; these counts are not distinct-record counts or evaluation-set rule performance.</p>
<p>{link(population_path, f'Full population CSV (row {csv_row}, including header)')}</p></div></details>''')

    values = {
        "bars": "\n".join(bars), "table": "\n".join(table), "rules": "\n".join(rules),
        "predictions": "\n".join(predictions), "best_score": percent(best["balanced_accuracy"]),
        "single_score": f'{float(baseline["balanced_accuracy"]) * 100:.2f}%',
        "ensemble_score": f'{float(ensemble["balanced_accuracy"]) * 100:.2f}%',
        "ensemble_accuracy": percent(ensemble["accuracy"]), "difference": f"{difference * 100:.2f}",
        "single_cv_score": percent(chosen["balanced_accuracy"]), "ensemble_cv_score": percent(cv_ensemble["balanced_accuracy"]),
        "single_detected": baseline["true_positives"], "ensemble_detected": ensemble["true_positives"],
        "single_missed": baseline["false_negatives"], "ensemble_missed": ensemble["false_negatives"],
        "single_false": baseline["false_positives"], "ensemble_false": ensemble["false_positives"],
        "single_recall": percent(baseline["recall"]), "ensemble_recall": percent(ensemble["recall"]),
        "single_precision": percent(baseline["precision"]), "ensemble_precision": percent(ensemble["precision"]),
        "iterations": f"{iterations:,}", "iterations_raw": str(iterations), "population": f"{population:,}",
        "training_rows": f"{training_rows:,}", "training_rows_raw": str(training_rows),
        "passes": f"{passes:g}", "cv_passes": f"{iterations / max(cv_training_rows):.1f}",
        "cv_training_min": f"{min(cv_training_rows):,}", "cv_training_max": f"{max(cv_training_rows):,}",
        "majority_accuracy": percent(majority_accuracy), "majority_accuracy_raw": repr(majority_accuracy),
    }
    return Template(TEMPLATE.read_text(encoding="utf-8")).substitute(values)


def main():
    page = build_page()
    output = ROOT / "results.html"
    output.write_text(page, encoding="utf-8", newline="\n")
    print(f"Built {output} from verified exploratory evidence. No models were trained or evaluated.")


if __name__ == "__main__":
    main()
