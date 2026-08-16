import pandas as pd

# Load the raw UCI Superconductivity training dataset
train_df = pd.read_csv("data/raw/train.csv")

# Keep an untouched copy of the raw dataset
train_original = train_df.copy(deep=True)

# Work on a copy so the original data is unchanged
train_transformed = train_original.copy(deep=True)

# Create the binary classification target
train_transformed["above_77k"] = (
    train_transformed["critical_temp"] > 77
).astype(int)

# Check the number of records in each class
print("Number of records in each 77 K class:")
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
