"""Run a small first eLCS smoke test on the raw dataset."""

from skeLCS import eLCS
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

if __package__:
    from .model_data import load_model_data
else:
    from model_data import load_model_data

RANDOM_STATE = 42
LEARNING_ITERATIONS = 100
POPULATION_SIZE = 100


def main() -> None:
    """Fit unmodified eLCS once and print its balanced accuracy."""
    X, y, groups = load_model_data()
    X_values = X.to_numpy()
    y_values = y.to_numpy()
    group_values = groups.to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_indices, test_indices = next(
        splitter.split(X_values, y_values, groups=group_values)
    )

    model = eLCS(
        learning_iterations=LEARNING_ITERATIONS,
        N=POPULATION_SIZE,
        random_state=RANDOM_STATE,
    )
    model.fit(X_values[train_indices], y_values[train_indices])
    predictions = model.predict(X_values[test_indices])

    score = balanced_accuracy_score(y_values[test_indices], predictions)
    print(f"Training rows: {len(train_indices)}")
    print(f"Test rows: {len(test_indices)}")
    print(f"Balanced accuracy: {score:.3f}")


if __name__ == "__main__":
    main()
