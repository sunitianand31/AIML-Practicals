# AIML Assignment 1
# Association Between Independent and Dependent Variables

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load external dataset
df = pd.read_csv("student-mat.csv", sep=None, engine="python")

# Step 2: Display dataset
print("First Five Rows of Dataset:")
print(df.head())

print("\nDataset Columns:")
print(df.columns.tolist())

# Step 3: Select variables
X = df["studytime"]
Y = df["G3"]

print("\nIndependent Variable: Study Time")
print("Dependent Variable: Final Grade (G3)")

# Step 4: Calculate correlation
correlation = X.corr(Y)

print("\nCorrelation Coefficient:", round(correlation, 3))

if correlation > 0:
    print("Positive Association")
elif correlation < 0:
    print("Negative Association")
else:
    print("No Linear Association")

# Step 5: Plot the relationship
plt.figure(figsize=(8, 5))

sns.regplot(x=X, y=Y)

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nAssignment 1 completed successfully!")