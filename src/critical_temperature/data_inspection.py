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

#Check feature variance
print("\n--- FEATURE VARIANCE ---")

train_numeric = train_original.select_dtypes(include="number")
feature_variance = train_numeric.var().sort_values()

print(feature_variance.to_string())


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


# Inspect missing values
print("\n--- TRAIN.CSV MISSING VALUES ---")
print(train_df.isna().sum().to_string())

print("\n--- UNIQUE_M.CSV MISSING VALUES ---")
print(unique_df.isna().sum().to_string())


# Inspect exact duplicate rows
print("\n--- DUPLICATE ROWS ---")
print("train.csv duplicate rows:", train_df.duplicated().sum())
print("unique_m.csv duplicate rows:", unique_df.duplicated().sum())


# Inspect infinite values
import numpy as np

print("\n--- INFINITE VALUES ---")
print("train.csv infinite values:",
      np.isinf(train_df.select_dtypes(include="number")).sum().sum())

print("unique_m.csv infinite values:",
      np.isinf(unique_df.select_dtypes(include="number")).sum().sum())


# Check critical temperature consistency between the two files
temp_mismatches = (
    train_df["critical_temp"] != unique_df["critical_temp"]
).sum()

print("\n--- CRITICAL TEMPERATURE CONSISTENCY ---")
print("Temperature mismatches between files:", temp_mismatches)


# Check formatting of material names
print("\n--- MATERIAL FORMATTING ---")
print("Blank material names:",
      unique_df["material"].astype(str).str.strip().eq("").sum())

print("Material names with leading/trailing spaces:",
      (unique_df["material"] != unique_df["material"].str.strip()).sum())

# Check numerical columns for potential outliers using the IQR method
print("\n--- POTENTIAL OUTLIERS ---")

numeric_columns = train_original.select_dtypes(include="number").columns

outlier_counts = {}

for column in numeric_columns:
    Q1 = train_original[column].quantile(0.25)
    Q3 = train_original[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (train_original[column] < lower_bound) |
        (train_original[column] > upper_bound)
    ).sum()

    outlier_counts[column] = outliers

outlier_counts = pd.Series(outlier_counts)

print(
    outlier_counts[outlier_counts > 0]
    .sort_values(ascending=False)
    .to_string()
)

# Review rows that are exact duplicates in train.csv
duplicate_mask = train_original.duplicated(keep=False)

duplicate_review = unique_original.loc[
    duplicate_mask,
    ["material", "critical_temp"]
]

print("\n--- DUPLICATE REVIEW ---")
print("Rows involved:", len(duplicate_review))
print(duplicate_review.head(30).to_string())

print("\nUnique materials in duplicated rows:")
print(duplicate_review["material"].nunique())

print("\nRepeated material names:")
print(duplicate_review["material"].duplicated().sum())