"""Preprocess the superconductivity dataset for Phase II."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_preprocessed.csv"


def preprocess_data():
    """Load, clean, and save the preprocessed dataset."""

    data = pd.read_csv(RAW_DATA_PATH)

    print(f"Original rows: {len(data)}")
    print(f"Original columns: {len(data.columns)}")


    duplicate_count = data.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")

    data = data.drop_duplicates().reset_index(drop=True)



    data = data.dropna(subset=["critical_temp"])


    data = data[data["critical_temp"] > 0]


    numeric_columns = data.select_dtypes(include="number").columns

    data[numeric_columns] = data[numeric_columns].replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    data = data.dropna().reset_index(drop=True)



    data["above_77k"] = (
        data["critical_temp"] > 77
    ).astype(int)



    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print(f"Final rows: {len(data)}")
    print(f"Final columns: {len(data.columns)}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_data()