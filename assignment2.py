# AIML Assignment 2
# Aim: Apply Clustering Using Unsupervised Learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load External Dataset
df = pd.read_csv(
    "Wholesale customers data.csv",
    encoding="latin1"
)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("First Five Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 2: Select Numerical Features
features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df[features]

# Step 3: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply K-Means Clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Step 5: Display Cluster Information
print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())

print("\nFirst Five Rows with Cluster Labels:")
print(df[features + ["Cluster"]].head())

# Step 6: Visualize Clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Fresh"],
    df["Grocery"],
    c=df["Cluster"],
    cmap="viridis",
    s=50,
    alpha=0.7
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Fresh Product Spending")
plt.ylabel("Grocery Spending")
plt.grid(True)
plt.tight_layout()

# Save graph as an image
plt.savefig("assignment2_graph.png", dpi=300)

# Display graph
plt.show()

# Step 7: Save Clustering Results
df.to_csv("assignment2_clustered_output.csv", index=False)

print("\nGraph saved as assignment2_graph.png")
print("Clustered dataset saved as assignment2_clustered_output.csv")
print("Assignment 2 completed successfully!")