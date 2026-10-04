"""Build an offline results page from completed evidence; never fit or score models."""

import ast
import csv
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[2]
FINAL = Path("reports/holdout/final_comparison")
TEMPLATE = Path(__file__).with_name("results_page.html")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(root, name):
    path = (root / name.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe evidence path: {name}")
    return path


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
    """Validate the saved hash chain and return deterministic, self-contained HTML."""
    root = Path(root)
    folder = root / FINAL
    complete = read_json(folder / "evaluation_complete.json")
    check_hash(folder / "plan.json", complete["plan_sha256"])
    for name, digest in complete["outputs"].items():
        check_hash(safe_path(folder, name), digest)
    plan = read_json(folder / "plan.json")
    for entry in plan["experiments"]:
        experiment = safe_path(root, entry["directory"])
        check_hash(experiment / "config.json", entry["config_sha256"])
        check_hash(experiment / "cv_complete.json", entry["cv_completion_sha256"])
        check_hash(experiment / "test_complete.json", complete["final_manifests"][entry["directory"]])
        for name, digest in read_json(experiment / "test_complete.json").items():
            check_hash(safe_path(experiment, name), digest)
    rows = read_csv(folder / "test_comparison.csv")
    stats = read_json(folder / "statistical_test.json")
    baseline = next(row for row in rows if row["experiment"].replace("\\", "/") == plan["primary_comparison"]["baseline"])
    ensemble = next(row for row in rows if row["experiment"].replace("\\", "/") == plan["primary_comparison"]["improved"])
    best = max(rows, key=lambda row: float(row["balanced_accuracy"]))
    if len(rows) != 7 or best["model"] != "random_forest" or stats["difference"] >= 0:
        raise ValueError("This presentation describes the frozen seven-system study, not a new study.")

    bars, table, predictions = [], [], []
    for row in sorted(rows, key=lambda row: float(row["balanced_accuracy"]), reverse=True):
        score = float(row["balanced_accuracy"]) * 100
        style = "ensemble" if row["model"] == "ensemble_dedup" else "single" if row is baseline else ""
        name = escape(row["display_name"])
        bars.append(f'<li class="chart-row {style}" data-score="{score!r}"><div class="bar-label"><span>{name}</span><strong>{score:.1f}%</strong></div><div class="track" aria-hidden="true"><div class="bar" style="width:{score!r}%"></div></div></li>')
        table.append(f'<tr><th scope="row">{name}</th>' + ''.join(
            f'<td data-label="{label}">{percent(row[field])}</td>'
            for field, label in (("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))) + '</tr>')
        predictions.append('<li>' + link(Path(row["experiment"].replace("\\", "/")) / "test_predictions.csv", row["display_name"]) + '</li>')

    rules = []
    for seed, csv_row, fields in (
        (11, 23, ("mean_atomic_mass", "wtd_mean_Valence", "range_Valence")),
        (42, 76, ("range_atomic_mass", "mean_fie", "wtd_mean_Valence")),
        (73, 73, ("mean_Valence", "wtd_mean_atomic_radius", "mean_atomic_mass")),
    ):
        population = Path(plan["primary_comparison"]["improved"]) / f"rules/member_{seed}/rules.csv"
        row = read_csv(root / population)[csv_row - 2]
        names = row["Specified Attribute Names"].split(", ")
        values = ast.literal_eval("[" + row["Specified Values"] + "]")
        conditions = dict(zip(names, values, strict=True))
        excerpts = []
        for field in fields:
            value = conditions[field]
            condition = f"between {value[0]:.4f} and {value[1]:.4f} (strict)" if isinstance(value, list) else f"equal to {value:g}"
            excerpts.append(f'<li><code>{escape(field)}</code>: {condition}</li>')
        positive = int(float(row["above_77k"])) == 1
        predicted = "above 77 K" if positive else "at or below 77 K"
        rules.append(f'''<details class="rule-example"><summary>Seed {seed}: predicts {predicted}<span>{int(row["Correct Count"])} correct / {int(row["Match Count"])} training matches</span></summary>
<div class="disclosure"><p>This rule specifies <strong>{len(names)} of 81 features</strong>. Three condition excerpts:</p><ul class="conditions">{''.join(excerpts)}</ul>
<p>All specified conditions must match together. These rounded excerpts are not the complete rule. The remaining features are unrestricted.</p>
<p>Training accuracy: <strong>{percent(row["Accuracy"])}</strong>. Fitness: {float(row["Fitness"]):.5f}. Numerosity: {int(row["Numerosity"])}. This is limited training experience, not held-out rule performance or a causal explanation.</p>
<p>{link(population, f'Full population CSV (row {csv_row}, including header)')}</p></div></details>''')

    values = {
        "bars": "\n".join(bars), "table": "\n".join(table), "rules": "\n".join(rules),
        "predictions": "\n".join(predictions), "best_score": percent(best["balanced_accuracy"]),
        "single_score": percent(baseline["balanced_accuracy"]), "ensemble_score": percent(ensemble["balanced_accuracy"]),
        "difference": f'{abs(stats["difference"]) * 100:.1f}', "p_value": f'{stats["p_value"]:.5f}',
        "single_detected": baseline["true_positives"], "ensemble_detected": ensemble["true_positives"],
        "single_missed": baseline["false_negatives"], "ensemble_missed": ensemble["false_negatives"],
        "single_false": baseline["false_positives"], "ensemble_false": ensemble["false_positives"],
        "single_recall": percent(baseline["recall"]), "ensemble_recall": percent(ensemble["recall"]),
        "single_precision": percent(baseline["precision"]), "ensemble_precision": percent(ensemble["precision"]),
        "groups": f'{stats["group_count"]:,}', "informative": str(stats["informative_groups"]),
        "draws": f'{stats["draws"]:,}',
    }
    return Template(TEMPLATE.read_text(encoding="utf-8")).substitute(values)


def main():
    page = build_page()
    output = ROOT / "results.html"
    output.write_text(page, encoding="utf-8", newline="\n")
    print(f"Built {output} from verified saved evidence. No models were trained or evaluated.")


if __name__ == "__main__":
    main()
