import pandas as pd

# Load the raw UCI Superconductivity datasets
train_df = pd.read_csv("data/raw/train.csv")
unique_df = pd.read_csv("data/raw/unique_m.csv")

print("Datasets loaded successfully.")

print("\ntrain.csv shape:", train_df.shape)
print("unique_m.csv shape:", unique_df.shape)