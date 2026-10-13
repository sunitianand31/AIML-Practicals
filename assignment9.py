# AIML Assignment 9
# Aim: Implement Support Vector Machine (SVM) Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Step 1: Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("First Five Rows of Dataset:")
print(X.head())

print("\nDataset Shape:", X.shape)

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# Step 3: Standardize Features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Create and Train SVM Model
model = SVC(kernel="linear")

model.fit(X_train_scaled, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("\n--- SVM Classification Results ---")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))

# Step 7: Display Confusion Matrix
display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=data.target_names,
    cmap="Blues"
)

plt.title("SVM Classification - Confusion Matrix")
plt.tight_layout()

# Step 8: Save Graph
plt.savefig("assignment9_svm_confusion_matrix.png", dpi=300)
plt.show()

# Step 9: Save Predictions
results = pd.DataFrame({
    "Actual": [data.target_names[i] for i in y_test],
    "Predicted": [data.target_names[i] for i in y_pred]
})

results.to_csv("assignment9_svm_predictions.csv", index=False)

print("\nGraph saved as assignment9_svm_confusion_matrix.png")
print("Predictions saved as assignment9_svm_predictions.csv")
print("Assignment 9 completed successfully!")