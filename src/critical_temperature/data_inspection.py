import pandas as pd

# Load the raw UCI Superconductivity datasets
train_df = pd.read_csv("data/raw/train.csv")
unique_df = pd.read_csv("data/raw/unique_m.csv")

# Keep untouched copies of the raw datasets
train_original = train_df.copy(deep=True)
unique_original = unique_df.copy(deep=True)

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

# Work on a copy so the original data is unchanged
train_transformed = train_original.copy(deep=True)

# Create the binary classification target
train_transformed["above_77k"] = (
    train_transformed["critical_temp"] > 77
).astype(int)

# Check the number of records in each class
print("\nNumber of records in each 77 K class:")
print(
    train_transformed["above_77k"]
    .value_counts()
    .sort_index()
)

# Check the proportion of records in each class
print("\nProportion of records in each 77 K class:")
print(
    train_transformed["above_77k"]
    .value_counts(normalize=True)
    .sort_index()
)
