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

# Create the output directory before saving cleaned datasets
Path("data/processed").mkdir(parents=True, exist_ok=True)
train_clean.to_csv("data/processed/train_clean.csv", index=False)
unique_clean.to_csv("data/processed/unique_m_clean.csv", index=False)

print("\nCleaned datasets saved successfully.")