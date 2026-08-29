import pandas as pd

# Load transformed dataset
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    project_root / "data" / "processed" / "train_transformed.csv"
)

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# Descriptive statistics
print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe().T.to_string())

# Class distribution
print("\n--- CLASS DISTRIBUTION ---")
print(df["above_77k"].value_counts().sort_index())

print("\nClass proportions:")
print(df["above_77k"].value_counts(normalize=True).sort_index())

import matplotlib.pyplot as plt

# Plot class distribution
df["above_77k"].value_counts().sort_index().plot(kind="bar")

plt.title("Class Distribution")
plt.xlabel("Above 77 K")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["0 = 77 K or below", "1 = Above 77 K"], rotation=0)
plt.tight_layout()
plt.show()

# Histogram of critical temperature
plt.hist(df["critical_temp"], bins=30)

plt.title("Distribution of Critical Temperature")
plt.xlabel("Critical Temperature (K)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot of critical temperature
plt.boxplot(df["critical_temp"])

plt.title("Critical Temperature Boxplot")
plt.ylabel("Critical Temperature (K)")
plt.tight_layout()
plt.show()

# Correlation analysis
correlations = (
    df.corr(numeric_only=True)["critical_temp"]
    .drop(["critical_temp", "above_77k"])
    .abs()
    .sort_values(ascending=False)
)

print("\n--- TOP CORRELATIONS WITH CRITICAL TEMPERATURE ---")
print(correlations.head(10))

# Select the 10 most correlated features
top_features = correlations.head(10).index.tolist()
heatmap_columns = top_features + ["critical_temp"]

correlation_matrix = df[heatmap_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(heatmap_columns)),
    heatmap_columns,
    rotation=90
)

plt.yticks(
    range(len(heatmap_columns)),
    heatmap_columns
)

plt.title("Correlation Heatmap of Key Features")
plt.tight_layout()
plt.show()

# Scatter plot: weighted mean valence vs critical temperature
plt.scatter(
    df["wtd_mean_Valence"],
    df["critical_temp"],
    alpha=0.4
)

plt.title("Weighted Mean Valence vs Critical Temperature")
plt.xlabel("Weighted Mean Valence")
plt.ylabel("Critical Temperature (K)")
plt.tight_layout()
plt.show()

print("\n--- KEY EDA FINDINGS ---")
print("The dataset is class imbalanced, with fewer materials above 77 K.")
print("Critical temperature is unevenly distributed and has one clear high outlier.")
print("Several thermal conductivity and atomic-radius features are related to critical temperature.")
print("Weighted mean valence shows a negative relationship with critical temperature.")