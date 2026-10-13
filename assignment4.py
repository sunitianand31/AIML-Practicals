# AIML Assignment 4
# Aim: Implement Naive Bayes Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load External Iris Dataset
columns = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    "Species"
]

df = pd.read_csv(
    "iris.data",
    header=None,
    names=columns
)

# Remove any empty rows
df.dropna(inplace=True)

print("First Five Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 2: Separate Features and Target
X = df.drop("Species", axis=1)
y = df["Species"]

# Step 3: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", X_train.shape[0])
print("Testing Data Size:", X_test.shape[0])

# Step 4: Train Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 7: Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot()
plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()

# Save graph
plt.savefig("assignment4_confusion_matrix.png", dpi=300)
plt.show()

# Step 8: Save Predictions
results = X_test.copy()
results["Actual Species"] = y_test
results["Predicted Species"] = y_pred

results.to_csv("assignment4_predictions.csv", index=False)

print("\nGraph saved as assignment4_confusion_matrix.png")
print("Predictions saved as assignment4_predictions.csv")
print("Assignment 4 completed successfully!")