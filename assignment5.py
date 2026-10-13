# AIML Assignment 5
# Aim: Implement K-Nearest Neighbors (KNN) Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Define Dataset Columns
columns = [
    "ID", "Diagnosis",
    "Radius_mean", "Texture_mean", "Perimeter_mean",
    "Area_mean", "Smoothness_mean", "Compactness_mean",
    "Concavity_mean", "Concave_points_mean",
    "Symmetry_mean", "Fractal_dimension_mean",
    "Radius_se", "Texture_se", "Perimeter_se",
    "Area_se", "Smoothness_se", "Compactness_se",
    "Concavity_se", "Concave_points_se",
    "Symmetry_se", "Fractal_dimension_se",
    "Radius_worst", "Texture_worst", "Perimeter_worst",
    "Area_worst", "Smoothness_worst", "Compactness_worst",
    "Concavity_worst", "Concave_points_worst",
    "Symmetry_worst", "Fractal_dimension_worst"
]

# Step 2: Load External Dataset
df = pd.read_csv(
    "wdbc.data",
    header=None,
    names=columns
)

print("First Five Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 3: Separate Features and Target
X = df.drop(["ID", "Diagnosis"], axis=1)
y = df["Diagnosis"]

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", X_train.shape[0])
print("Testing Data Size:", X_test.shape[0])

# Step 5: Standardize the Features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Apply KNN Classification
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)

print("\nKNN Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 9: Display Confusion Matrix
labels = ["B", "M"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)

display.plot()
plt.title("KNN Classification - Confusion Matrix")
plt.tight_layout()

plt.savefig("assignment5_confusion_matrix.png", dpi=300)
plt.show()

# Step 10: Save Predictions
results = X_test.copy()
results["Actual Diagnosis"] = y_test
results["Predicted Diagnosis"] = y_pred

results.to_csv("assignment5_predictions.csv", index=False)

print("\nGraph saved as assignment5_confusion_matrix.png")
print("Predictions saved as assignment5_predictions.csv")
print("Assignment 5 completed successfully!")