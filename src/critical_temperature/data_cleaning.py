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

# Remove exact duplicate records
train_duplicates = train_clean.duplicated().sum()
unique_duplicates = unique_clean.duplicated().sum()

print("\n--- EXACT DUPLICATES ---")
print("train.csv duplicates:", train_duplicates)
print("unique_m.csv duplicates:", unique_duplicates)

train_clean = train_clean.drop_duplicates()
unique_clean = unique_clean.drop_duplicates()

print("\nShapes after duplicate removal:")
print("train.csv:", train_clean.shape)
print("unique_m.csv:", unique_clean.shape)