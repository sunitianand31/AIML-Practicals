# AIML Assignment 8
# Aim: Implement Regularization using Ridge and Lasso Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load External Dataset
df = pd.read_csv("student-mat.csv", sep=",")

print("First Five Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 2: Select Features and Target
features = [
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]

X = df[features]
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

# Step 4: Standardize Features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

ridge_pred = ridge_model.predict(X_test_scaled)

# Step 6: Train Lasso Regression
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)

lasso_pred = lasso_model.predict(X_test_scaled)

# Step 7: Evaluate Both Models
print("\n--- Ridge Regression Results ---")
print("MAE:", round(mean_absolute_error(y_test, ridge_pred), 2))
print("MSE:", round(mean_squared_error(y_test, ridge_pred), 2))
print("R2 Score:", round(r2_score(y_test, ridge_pred), 2))

print("\n--- Lasso Regression Results ---")
print("MAE:", round(mean_absolute_error(y_test, lasso_pred), 2))
print("MSE:", round(mean_squared_error(y_test, lasso_pred), 2))
print("R2 Score:", round(r2_score(y_test, lasso_pred), 2))

# Step 8: Compare Actual and Predicted Values
comparison = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Ridge Predicted Marks": ridge_pred.round(2),
    "Lasso Predicted Marks": lasso_pred.round(2)
})

print("\nFirst Five Prediction Comparisons:")
print(comparison.head())

# Step 9: Visualize Results
plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    ridge_pred,
    label="Ridge Regression",
    alpha=0.7
)

plt.scatter(
    y_test,
    lasso_pred,
    label="Lasso Regression",
    marker="x",
    alpha=0.7
)

plt.plot(
    [0, 20],
    [0, 20],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Ridge vs Lasso Regression")
plt.xlabel("Actual Final Grades")
plt.ylabel("Predicted Final Grades")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save Graph
plt.savefig("assignment8_regularization_graph.png", dpi=300)
plt.show()

# Step 10: Save Results
comparison.to_csv("assignment8_predictions.csv", index=False)

print("\nGraph saved as assignment8_regularization_graph.png")
print("Predictions saved as assignment8_predictions.csv")
print("Assignment 8 completed successfully!")