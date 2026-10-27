# ==========================================
# 1. Import Required Libraries
# ==========================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 2. Load the Dataset
# ==========================================
data = pd.read_csv(
    r"C:\Users\staff\Downloads\archive(1)\temperatures.csv"
)

df = data.copy()

# ==========================================
# 3. Explore the Dataset
# ==========================================
print("First Five Records:")
print(data.head())

print("\nLast Five Records:")
print(data.tail())

print("\nDataset Type:")
print(type(data))

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
data.info()

print("\nSummary Statistics:")
print(data.describe())

# ==========================================
# 4. Check Missing Values
# ==========================================
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

print("\nFirst Five Rows of Missing Value Check:")
print(data.isnull().head())

# Count the number of zero values in each column
print("\nZero Values in Each Column:")
print((data == 0).sum())

# Count occurrences of temperature 22 in JAN
if "JAN" in data.columns:
    count = (data["JAN"] == 22).sum()
    print("\nCount of JAN values equal to 22:", count)

# ==========================================
# 5. Prepare Input and Output Variables
# ==========================================
# Input variable
x = data[["YEAR"]]

# Output variable
y = data["ANNUAL"]

# Remove rows with missing input or output values
valid_rows = x["YEAR"].notna() & y.notna()

x = x.loc[valid_rows]
y = y.loc[valid_rows]

print("\nInput Shape:", x.shape)
print("Output Shape:", y.shape)

# ==========================================
# 6. Visualize Temperature vs Year
# ==========================================
plt.figure(figsize=(10, 5))

plt.plot(x["YEAR"], y, "o")

plt.xlabel("Year")
plt.ylabel("Annual Temperature")
plt.title("Temperature vs Year")
plt.grid(True)
plt.show()

# ==========================================
# 7. Scatter Plot Using Seaborn
# ==========================================
plt.figure(figsize=(10, 5))

sns.scatterplot(x=x["YEAR"], y=y)

plt.title("Temperature vs Year")
plt.xlabel("Year")
plt.ylabel("Annual Temperature")
plt.grid(True)
plt.show()

# ==========================================
# 8. Split the Dataset
# ==========================================
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining and Testing Dataset Shapes:")
print(f"X Training Dataset: {x_train.shape}")
print(f"Y Training Dataset: {y_train.shape}")
print(f"X Testing Dataset: {x_test.shape}")
print(f"Y Testing Dataset: {y_test.shape}")

# ==========================================
# 9. Train the Linear Regression Model
# ==========================================
model = LinearRegression()

model.fit(x_train, y_train)

print("\nModel Coefficient:")
print(model.coef_)

print("\nModel Intercept:")
print(model.intercept_)

# ==========================================
# 10. Make Predictions
# ==========================================
y_pred = model.predict(x_test)

print("\nPredicted Annual Temperatures:")
print(y_pred)

print("\nPrediction Shape:")
print(y_pred.shape)

# ==========================================
# 11. Plot Training Data and Regression Line
# ==========================================
plt.figure(figsize=(10, 5))

plt.scatter(
    x_train["YEAR"],
    y_train,
    label="Training Data"
)

# Sort years so the regression line displays correctly
sorted_indices = np.argsort(x_train["YEAR"].values)

plt.plot(
    x_train["YEAR"].values[sorted_indices],
    model.predict(x_train.iloc[sorted_indices]),
    color="red",
    linewidth=3,
    label="Regression Line"
)

plt.title("Linear Regression: Temperature vs Year")
plt.xlabel("Year")
plt.ylabel("Annual Temperature")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 12. Regression Plot Using Seaborn
# ==========================================
plt.figure(figsize=(10, 5))

sns.regplot(
    x=x_train["YEAR"],
    y=y_train
)

plt.title("Linear Regression: Temperature vs Year")
plt.xlabel("Year")
plt.ylabel("Annual Temperature")
plt.grid(True)
plt.show()

# ==========================================
# 13. Evaluate the Model
# ==========================================
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-Square (R²): {r2:.4f}")

# ==========================================
# 14. Compare Actual and Predicted Values
# ==========================================
results = pd.DataFrame({
    "Year": x_test["YEAR"].values,
    "Actual Temperature": y_test.values,
    "Predicted Temperature": y_pred
})

print("\nActual vs Predicted Temperatures:")
print(results.sort_values("Year").to_string(index=False))

# ==========================================
# 15. Final Output
# ==========================================
print("\n======================================")
print("Temperature Prediction Completed")
print("Linear Regression Model Trained Successfully")
print("======================================")