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