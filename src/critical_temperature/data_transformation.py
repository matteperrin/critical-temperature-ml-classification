import pandas as pd

# Load the cleaned UCI Superconductivity training dataset
train_df = pd.read_csv("data/processed/train_clean.csv")

# Keep an untouched copy of the cleaned dataset
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

# Save the transformed dataset
train_transformed.to_csv(
    "data/processed/train_transformed.csv",
    index=False
)

print("\nTransformed dataset saved successfully.")