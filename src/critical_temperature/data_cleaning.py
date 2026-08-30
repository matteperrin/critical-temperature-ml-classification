from pathlib import Path

import pandas as pd

# Load raw datasets
train_df = pd.read_csv("data/raw/train.csv")
unique_df = pd.read_csv("data/raw/unique_m.csv")

# Create copies for cleaning so raw data remains unchanged
train_clean = train_df.copy()
unique_clean = unique_df.copy()

print("Raw datasets loaded successfully.")
print("train.csv:", train_clean.shape)
print("unique_m.csv:", unique_clean.shape)

# Confirm the inspection findings before writing the cleaned datasets
train_numeric = train_clean.select_dtypes(include="number")
unique_numeric = unique_clean.select_dtypes(include="number")
element_columns = unique_clean.columns.drop(["critical_temp", "material"])
validation_checks = {
    "missing values": not train_clean.isna().any().any()
    and not unique_clean.isna().any().any(),
    "infinite values": not train_numeric.isin([float("inf"), float("-inf")]).any().any()
    and not unique_numeric.isin([float("inf"), float("-inf")]).any().any(),
    "file alignment": len(train_clean) == len(unique_clean)
    and train_clean["critical_temp"].equals(unique_clean["critical_temp"]),
    "invalid values": train_clean["number_of_elements"].gt(0).all()
    and train_clean["critical_temp"].gt(0).all()
    and unique_clean[element_columns].ge(0).all().all(),
    "material formatting": unique_clean["material"].str.strip().ne("").all()
    and unique_clean["material"].eq(unique_clean["material"].str.strip()).all(),
}
failed_checks = [name for name, passed in validation_checks.items() if not passed]
if failed_checks:
    raise ValueError("Data validation failed: " + ", ".join(failed_checks))

# Check exact duplicate records
train_duplicates = train_clean.duplicated().sum()
unique_duplicates = unique_clean.duplicated().sum()

print("\n--- EXACT DUPLICATES ---")
print("train.csv duplicates:", train_duplicates)
print("unique_m.csv duplicates:", unique_duplicates)

# Cleaning decisions
# No missing, infinite, invalid, or formatting issues were found.
# Potential outliers were retained because they may represent valid material properties.
# Duplicate rows in train.csv were retained because some correspond
# to different material records in unique_m.csv.

print("\nNo records were removed during cleaning.")
print("Clean train.csv shape:", train_clean.shape)
print("Clean unique_m.csv shape:", unique_clean.shape)

print("\nFinal dataset shapes:")
print("train.csv:", train_clean.shape)
print("unique_m.csv:", unique_clean.shape)

# Save processed copies; the original files in data/raw remain unchanged.
Path("data/processed").mkdir(parents=True, exist_ok=True)
train_clean.to_csv("data/processed/train_clean.csv", index=False)
unique_clean.to_csv("data/processed/unique_m_clean.csv", index=False)

print("\nCleaned datasets saved successfully.")