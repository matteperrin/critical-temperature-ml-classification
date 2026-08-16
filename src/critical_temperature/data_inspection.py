import pandas as pd

# Load the raw UCI Superconductivity datasets
train_df = pd.read_csv("data/raw/train.csv")
unique_df = pd.read_csv("data/raw/unique_m.csv")

print("Datasets loaded successfully.")

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