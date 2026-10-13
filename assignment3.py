# AIML Assignment 3
# Aim: Dimensionality Reduction using PCA

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load External Dataset
columns = [
    "Class", "Alcohol", "Malic_acid", "Ash",
    "Alcalinity_of_ash", "Magnesium", "Total_phenols",
    "Flavanoids", "Nonflavanoid_phenols",
    "Proanthocyanins", "Color_intensity", "Hue",
    "OD280_OD315", "Proline"
]

df = pd.read_csv(
    "wine.data",
    header=None,
    names=columns
)

print("First Five Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 2: Separate Features and Class
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 3: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Create Reduced Dataset
pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

pca_df["Class"] = y.values

print("\nDataset After PCA:")
print(pca_df.head())

print("\nOriginal Number of Features:", X.shape[1])
print("Reduced Number of Features:", X_pca.shape[1])

variance = sum(pca.explained_variance_ratio_) * 100
print("Variance Explained by Two Components:", round(variance, 2), "%")

# Step 6: Visualize PCA Results
plt.figure(figsize=(8, 6))

for wine_class in sorted(pca_df["Class"].unique()):
    subset = pca_df[pca_df["Class"] == wine_class]

    plt.scatter(
        subset["Principal Component 1"],
        subset["Principal Component 2"],
        label=f"Class {wine_class}",
        s=50,
        alpha=0.7
    )

plt.title("Wine Dataset - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Step 7: Save and Display Graph
plt.savefig("assignment3_pca_graph.png", dpi=300)
plt.show()

# Step 8: Save Results
pca_df.to_csv("assignment3_pca_output.csv", index=False)

print("\nGraph saved as assignment3_pca_graph.png")
print("PCA output saved as assignment3_pca_output.csv")
print("Assignment 3 completed successfully!")