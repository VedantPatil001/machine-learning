# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from statsmodels.stats.outliers_influence import variance_inflation_factor

print("All libraries imported successfully!")


# ============================================================
# 2. LOAD THE CALIFORNIA HOUSING DATASET
# ============================================================
housing = fetch_california_housing(as_frame=True)

X = housing.data.copy()
y = housing.target.copy()

y.name = "MedianHouseValue"

print("\nDataset Shape:", X.shape)

print("\nFeatures:")
print(list(X.columns))

print("\nFirst Five Records:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

print("\nDataset Information:")
X.info()

print("\nSummary Statistics:")
print(X.describe())

print("\nMissing Values:")
print(X.isnull().sum())


# ============================================================
# 3. SPLIT THE DATASET INTO TRAINING AND TESTING SETS
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================================
# 4. VARIANCE INFLATION FACTOR (VIF) ANALYSIS
# ============================================================
vif_scaler = StandardScaler()

X_train_scaled = vif_scaler.fit_transform(X_train)

vif_data = pd.DataFrame({
    "Feature": X_train.columns,
    "VIF": [
        variance_inflation_factor(X_train_scaled, i)
        for i in range(X_train_scaled.shape[1])
    ]
})

print("\nVariance Inflation Factor (VIF):")

print(
    vif_data.sort_values(
        by="VIF",
        ascending=False
    ).round(3).to_string(index=False)
)


# ============================================================
# 5. DEFINE MODEL EVALUATION FUNCTION
# ============================================================
def evaluate_model(
    model_name,
    y_train,
    train_pred,
    y_test,
    test_pred
):
    return {
        "Model": model_name,

        "Train MSE": mean_squared_error(
            y_train, train_pred
        ),

        "Test MSE": mean_squared_error(
            y_test, test_pred
        ),

        "Train RMSE": np.sqrt(
            mean_squared_error(y_train, train_pred)
        ),

        "Test RMSE": np.sqrt(
            mean_squared_error(y_test, test_pred)
        ),

        "Train MAE": mean_absolute_error(
            y_train, train_pred
        ),

        "Test MAE": mean_absolute_error(
            y_test, test_pred
        ),

        "Train R2": r2_score(
            y_train, train_pred
        ),

        "Test R2": r2_score(
            y_test, test_pred
        )
    }


results = []

print("\nEvaluation function created successfully!")


# ============================================================
# 6. LINEAR REGRESSION
# ============================================================
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_train_pred = linear_model.predict(X_train)
linear_test_pred = linear_model.predict(X_test)

results.append(
    evaluate_model(
        "Linear Regression",
        y_train,
        linear_train_pred,
        y_test,
        linear_test_pred
    )
)

print("\nLinear Regression completed successfully!")


# ============================================================
# 7. RIDGE REGRESSION WITH GRIDSEARCHCV
# ============================================================
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

ridge_params = {
    "model__alpha": np.logspace(-4, 4, 50)
}

ridge_grid = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid=ridge_params,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

ridge_grid.fit(X_train, y_train)

best_ridge = ridge_grid.best_estimator_

ridge_train_pred = best_ridge.predict(X_train)
ridge_test_pred = best_ridge.predict(X_test)

results.append(
    evaluate_model(
        "Ridge Regression",
        y_train,
        ridge_train_pred,
        y_test,
        ridge_test_pred
    )
)

print("\nBest Ridge Parameter:")
print(ridge_grid.best_params_)

print("Ridge Regression completed successfully!")


# ============================================================
# 8. LASSO REGRESSION WITH GRIDSEARCHCV
# ============================================================
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=10000))
])

lasso_params = {
    "model__alpha": np.logspace(-4, 1, 50)
}

lasso_grid = GridSearchCV(
    estimator=lasso_pipeline,
    param_grid=lasso_params,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

lasso_grid.fit(X_train, y_train)

best_lasso = lasso_grid.best_estimator_

lasso_train_pred = best_lasso.predict(X_train)
lasso_test_pred = best_lasso.predict(X_test)

results.append(
    evaluate_model(
        "Lasso Regression",
        y_train,
        lasso_train_pred,
        y_test,
        lasso_test_pred
    )
)

print("\nBest Lasso Parameter:")
print(lasso_grid.best_params_)

print("Lasso Regression completed successfully!")


# ============================================================
# 9. COMPARE MODEL PERFORMANCE
# ============================================================
results_df = pd.DataFrame(results)

print("\nPerformance Comparison:")

print(
    results_df.round(4).to_string(index=False)
)


# ============================================================
# 10. COMPARE MODEL COEFFICIENTS
# ============================================================
coefficient_df = pd.DataFrame({
    "Feature": X.columns,

    "Linear": linear_model.named_steps["model"].coef_,

    "Ridge": best_ridge.named_steps["model"].coef_,

    "Lasso": best_lasso.named_steps["model"].coef_
})

print("\nCoefficient Comparison:")

print(
    coefficient_df.round(4).to_string(index=False)
)


# ============================================================
# 11. LASSO FEATURE SELECTION
# ============================================================
lasso_coef = pd.Series(
    best_lasso.named_steps["model"].coef_,
    index=X.columns
)

print("\nLasso Coefficients:")
print(lasso_coef.round(4))

tolerance = 1e-8

selected_features = lasso_coef[
    np.abs(lasso_coef) > tolerance
].index.tolist()

removed_features = lasso_coef[
    np.abs(lasso_coef) <= tolerance
].index.tolist()

print("\nFeatures Selected by Lasso:")
print(selected_features)

print("\nFeatures Removed by Lasso:")
print(removed_features)


# ============================================================
# 12. VISUALIZE TRAINING VS TESTING R-SQUARED
# ============================================================
results_df.set_index("Model")[
    ["Train R2", "Test R2"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Training vs Testing R²")
plt.ylabel("R² Score")
plt.xlabel("Regression Model")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 13. VISUALIZE MODEL COEFFICIENTS
# ============================================================
coefficient_df.set_index("Feature").plot(
    kind="bar",
    figsize=(13, 6)
)

plt.title("Linear vs Ridge vs Lasso Coefficients")
plt.ylabel("Coefficient (Standardized Features)")
plt.xlabel("California Housing Feature")
plt.xticks(rotation=45, ha="right")
plt.axhline(y=0, linestyle="--")
plt.tight_layout()
plt.show()


# ============================================================
# 14. IDENTIFY THE BEST MODEL
# ============================================================
best_model_row = results_df.loc[
    results_df["Test MSE"].idxmin()
]

print("\n========================================")
print("BEST MODEL BASED ON TEST MSE")
print("========================================")

print("Model:", best_model_row["Model"])
print("Test MSE:", round(best_model_row["Test MSE"], 4))
print("Test RMSE:", round(best_model_row["Test RMSE"], 4))
print("Test MAE:", round(best_model_row["Test MAE"], 4))
print("Test R²:", round(best_model_row["Test R2"], 4))

print("\n========================================")
print("CALIFORNIA HOUSING PRICE PREDICTION")
print("COMPLETED SUCCESSFULLY")
print("========================================")