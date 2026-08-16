import pandas as pd

# Load the raw UCI Superconductivity datasets
train_df = pd.read_csv("data/raw/train.csv")
unique_df = pd.read_csv("data/raw/unique_m.csv")

# Keep untouched copies of the raw datasets
train_original = train_df.copy(deep=True)
unique_original = unique_df.copy(deep=True)

print("Datasets loaded successfully.")

# Essential rubric checks
print("\ntrain.csv shape:", train_df.shape)
print("unique_m.csv shape:", unique_df.shape)

#Inspect data types
print("\n--- TRAIN.CSV DATA TYPES ---")
print(train_df.dtypes.to_string())

print("\n--- UNIQUE_M.CSV DATA TYPES ---")
print(unique_df.dtypes.to_string())

#Inspect value ranges
print("\n--- TRAIN.CSV VALUE RANGES ---")
print(train_df.describe().T[["min", "max"]].to_string())

print("\n--- UNIQUE_M.CSV VALUE RANGES ---")
print(unique_df.describe().T[["min", "max"]].to_string())

#Inspect number of unique values
print("\n--- TRAIN.CSV UNIQUE VALUE COUNTS ---")
print(train_original.nunique(dropna=False).sort_values().to_string())

print("\n--- UNIQUE_M.CSV UNIQUE VALUE COUNTS ---")
print(unique_original.nunique(dropna=False).sort_values().to_string())

# Check missing values in train.csv
missing = train_original.isna().sum()

print("\nColumns containing missing values in train.csv:")
print(
    missing[missing > 0]
    .sort_values(ascending=False)
)

# Check duplicate rows in train.csv
print("\nNumber of duplicate rows in train.csv:")
print(train_original.duplicated().sum())

# Check missing values in unique_m.csv
missing = unique_original.isna().sum()

print("\nColumns containing missing values in unique_m.csv:")
print(
    missing[missing > 0]
    .sort_values(ascending=False)
)

# Check duplicate rows in unique_m.csv
print("\nNumber of duplicate rows in unique_m.csv:")
print(unique_original.duplicated().sum())

# Check blank and inconsistent material labels
print("\nBlank material labels in unique_m.csv:")
print(unique_original["material"].str.strip().eq("").sum())

print("\nMaterial labels with surrounding spaces:")
print(
    (
        unique_original["material"]
        != unique_original["material"].str.strip()
    ).sum()
)

# Check infinite numeric values
train_numeric = train_original.select_dtypes(include="number")
unique_numeric = unique_original.select_dtypes(include="number")

print("\nInfinite numeric values in train.csv:")
print(
    train_numeric
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print("\nInfinite numeric values in unique_m.csv:")
print(
    unique_numeric
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

# Check values that are invalid for these documented variables
print("\nNon-positive number_of_elements values:")
print((train_original["number_of_elements"] <= 0).sum())

print("\nNon-positive critical_temp values in train.csv:")
print((train_original["critical_temp"] <= 0).sum())

print("\nNegative elemental quantities in unique_m.csv:")
element_columns = unique_original.columns.drop(
    ["critical_temp", "material"]
)
print(
    (unique_original[element_columns] < 0)
    .sum()
    .sum()
)

# Check duplicated predictor values without using critical_temp
train_features = train_original.drop(columns=["critical_temp"])
print("\nRows with duplicated predictor values in train.csv:")
print(train_features.duplicated().sum())

# Review duplicated formulas and their recorded temperatures
formula_duplicates = unique_original.loc[
    unique_original["material"].duplicated(keep=False),
    ["material", "critical_temp"]
].sort_values(["material", "critical_temp"])

print("\nRecords containing duplicated chemical formulas:")
print(len(formula_duplicates))

print("\nNumber of duplicated chemical formulas:")
print(formula_duplicates["material"].nunique())

formula_temperature_counts = (
    formula_duplicates
    .groupby("material")["critical_temp"]
    .nunique()
)
print("\nDuplicated formulas with multiple temperatures:")
print((formula_temperature_counts > 1).sum())

# Supporting checks

# Zero values may be valid, so report them without removing them
zero_values = (train_numeric == 0).sum()
print("\nColumns containing zero values in train.csv:")
print(
    zero_values[zero_values > 0]
    .sort_values(ascending=False)
)

print("\nFirst 20 duplicated formula records for review:")
print(formula_duplicates.head(20).to_string(index=False))

# Confirm that the two source files align row by row
print("\nBoth datasets contain the same number of rows:")
print(len(train_original) == len(unique_original))

print("\ncritical_temp matches between the datasets:")
print(
    train_original["critical_temp"].equals(
        unique_original["critical_temp"]
    )
)
