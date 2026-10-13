# AIML Assignment 6
# Aim: Implement Linear Regression using an External Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("student-mat.csv", sep=",")

print("First Five Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 2: Select Input and Output Variables
X = df[["studytime"]]
y = df["G3"]

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Size:", X_train.shape[0])
print("Testing Data Size:", X_test.shape[0])

# Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):",
      round(mean_absolute_error(y_test, y_pred), 2))

print("Mean Squared Error (MSE):",
      round(mean_squared_error(y_test, y_pred), 2))

print("R2 Score:", round(r2_score(y_test, y_pred), 2))

print("\nSlope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))

# Step 7: Visualize Results
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test["studytime"],
    y_test,
    label="Actual Marks"
)

plt.scatter(
    X_test["studytime"],
    y_pred,
    marker="x",
    label="Predicted Marks"
)

# Regression Line
study_values = pd.DataFrame({
    "studytime": sorted(df["studytime"].unique())
})

plt.plot(
    study_values["studytime"],
    model.predict(study_values),
    label="Regression Line"
)

plt.title("Linear Regression: Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Step 8: Save Graph
plt.savefig("assignment6_regression_graph.png", dpi=300)
plt.show()

# Step 9: Save Predictions
results = X_test.copy()
results["Actual Marks"] = y_test
results["Predicted Marks"] = y_pred.round(2)

results.to_csv("assignment6_predictions.csv", index=False)

print("\nGraph saved as assignment6_regression_graph.png")
print("Predictions saved as assignment6_predictions.csv")
print("Assignment 6 completed successfully!")