"""Train eLCS and export its learned rule population for interpretation."""

import argparse
from pathlib import Path

import pandas as pd
from skeLCS import eLCS

if __package__:
    from .model_data import load_model_data
else:
    from model_data import load_model_data


RANDOM_STATE = 42
LEARNING_ITERATIONS = 1000
POPULATION_SIZE = 100

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def main(argv: list[str] | None = None) -> None:
    """Train eLCS and export learned classifier rules."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--iterations",
        type=int,
        default=LEARNING_ITERATIONS,
        help="Number of eLCS learning iterations.",
    )

    args = parser.parse_args(argv)

    if args.iterations <= 0:
        parser.error("--iterations must be a positive integer")

    X, y, _ = load_model_data()

    model = eLCS(
        learning_iterations=args.iterations,
        N=POPULATION_SIZE,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training eLCS for {args.iterations} iterations...",
        flush=True,
    )

    model.fit(
        X.to_numpy(),
        y.to_numpy(),
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rules_path = (
        REPORT_DIR
        / f"elcs_rules_{args.iterations}_iterations.csv"
    )

    model.export_final_rule_population(
        headerNames=X.columns.to_numpy(),
        className="above_77k",
        filename=rules_path,
        DCAL=True,
    )

    rules = pd.read_csv(rules_path)

    # Rank rules to make useful examples easier to inspect.
    ranked_rules = rules.sort_values(
        by=[
            "Accuracy",
            "Match Count",
            "Fitness",
        ],
        ascending=False,
    )

    top_rules_path = (
        REPORT_DIR
        / f"elcs_top_rules_{args.iterations}_iterations.csv"
    )

    ranked_rules.head(20).to_csv(
        top_rules_path,
        index=False,
    )

    print(f"Total rules exported: {len(rules)}")
    print(f"Full rule population saved to: {rules_path}")
    print(f"Top 20 rules saved to: {top_rules_path}")


if __name__ == "__main__":
    main()