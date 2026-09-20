"""Load model inputs without scaling, selecting features, or training models."""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data/raw/train.csv"


def load_model_data(path: str | Path = DEFAULT_DATA_PATH):
    """Load the data and return predictors, labels, and feature groups.

    This works with either the raw CSV or the Phase I transformed CSV. The
    target is always rebuilt from ``critical_temp``, so a stale saved label
    cannot affect the model inputs. Temperature and material names are left out
    of the predictors because they would cause target leakage or identify rows.

    Rows with identical feature values receive the same group ID. This helps
    with group-aware validation, but does not catch every repeated composition
    or related material family. The group IDs are created for this file only
    and should be regenerated whenever its rows change.
    """
    data = pd.read_csv(path)
    if data.empty or "critical_temp" not in data:
        raise ValueError("The data must be non-empty and include critical_temp.")

    X = data.drop(columns=["critical_temp", "above_77k", "material"], errors="ignore")
    numeric = pd.concat([X, data[["critical_temp"]]], axis=1)
    if X.empty or not all(is_numeric_dtype(dtype) for dtype in numeric.dtypes):
        raise ValueError("The predictors and critical_temp must be numeric.")
    if (
        numeric.isna().any().any()
        or numeric.isin([float("inf"), -float("inf")]).any().any()
    ):
        raise ValueError(
            "The predictors and critical_temp must be finite and non-missing."
        )
    if data["critical_temp"].le(0).any():
        raise ValueError("critical_temp must be positive, as required by Phase I.")

    y = data["critical_temp"].gt(77).astype(int).rename("above_77k")
    groups = X.groupby(list(X.columns), sort=False, dropna=False).ngroup()
    return X, y, groups.rename("feature_group")
