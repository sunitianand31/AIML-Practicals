# AIML Assignment 7
# Aim: Implement Support Vector Regression (SVR)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load External Dataset
df = pd.read_csv("student-mat.csv", sep=",")

print("First Five Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 2: Select Input and Target Variables
X = df[["studytime"]]
y = df["G3"]

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Size:", X_train.shape[0])
print("Testing Data Size:", X_test.shape[0])

# Step 4: Standardize Input Features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and Train SVR Model
model = SVR(kernel="rbf", C=100, epsilon=0.1)

model.fit(X_train_scaled, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate Model
print("\nSVR Model Evaluation:")

print("Mean Absolute Error (MAE):",
      round(mean_absolute_error(y_test, y_pred), 2))

print("Mean Squared Error (MSE):",
      round(mean_squared_error(y_test, y_pred), 2))

print("R2 Score:",
      round(r2_score(y_test, y_pred), 2))

# Step 8: Visualize Actual and Predicted Values
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test["studytime"],
    y_test,
    label="Actual Marks",
    alpha=0.7
)

plt.scatter(
    X_test["studytime"],
    y_pred,
    marker="x",
    label="Predicted Marks"
)

# Create smooth regression curve
study_values = pd.DataFrame({
    "studytime": sorted(df["studytime"].unique())
})

study_values_scaled = scaler.transform(study_values)
predicted_values = model.predict(study_values_scaled)

plt.plot(
    study_values["studytime"],
    predicted_values,
    label="SVR Regression Curve"
)

plt.title("Support Vector Regression: Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Step 9: Save Graph
plt.savefig("assignment7_svr_graph.png", dpi=300)
plt.show()

# Step 10: Save Predictions
results = X_test.copy()
results["Actual Marks"] = y_test
results["Predicted Marks"] = y_pred.round(2)

results.to_csv("assignment7_svr_predictions.csv", index=False)

print("\nGraph saved as assignment7_svr_graph.png")
print("Predictions saved as assignment7_svr_predictions.csv")
print("Assignment 7 completed successfully!")