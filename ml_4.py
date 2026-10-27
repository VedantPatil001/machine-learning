# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("All libraries imported successfully!")


# ============================================================
# 2. LOAD THE DATASET
# ============================================================
df = pd.read_csv(
    r"C:\Users\staff\Downloads\train_u6lujuX_CVtuZ9i.csv"
)

print("\nFirst Five Rows of the Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. REMOVE UNNECESSARY COLUMN
# ============================================================
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)

print("\nColumns After Removing Loan_ID:")
print(df.columns.tolist())


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================
# Remove rows where the target variable is missing
df = df.dropna(subset=["Loan_Status"]).copy()

# Input features
X = df.drop(columns=["Loan_Status"])

# Target variable: N = 0, Y = 1
y = df["Loan_Status"].map({
    "N": 0,
    "Y": 1
})

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nLoan Status Distribution:")
print(y.value_counts())


# ============================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 6. PREPROCESS NUMERICAL FEATURES
# ============================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ============================================================
# 7. PREPROCESS CATEGORICAL FEATURES
# ============================================================
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ============================================================
# 8. COMBINE PREPROCESSING STEPS
# ============================================================
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("\nPreprocessing pipeline created successfully!")


# ============================================================
# 9. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# 10. CREATE LOGISTIC REGRESSION MODEL
# ============================================================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nLogistic Regression Model Created Successfully!")


# ============================================================
# 11. TRAIN THE MODEL
# ============================================================
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================
y_pred = model.predict(X_test)

print("\nFirst Ten Actual Values:")
print(y_test.to_numpy()[:10])

print("\nFirst Ten Predicted Values:")
print(y_pred[:10])


# ============================================================
# 13. CALCULATE EVALUATION METRICS
# ============================================================
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========================================")
print("LOGISTIC REGRESSION MODEL EVALUATION")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# 14. DISPLAY CLASSIFICATION REPORT
# ============================================================
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Not Approved (N)",
            "Approved (Y)"
        ],
        zero_division=0
    )
)


# ============================================================
# 15. VISUALIZE CONFUSION MATRIX
# ============================================================
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Not Approved",
        "Approved"
    ],
    yticklabels=[
        "Not Approved",
        "Approved"
    ]
)

plt.title("Confusion Matrix - Loan Prediction")
plt.xlabel("Predicted Loan Status")
plt.ylabel("Actual Loan Status")

plt.tight_layout()
plt.show()


# ============================================================
# 16. DISPLAY PERFORMANCE SUMMARY
# ============================================================
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

print("\nPerformance Summary:")
print(metrics.to_string(index=False))


# ============================================================
# 17. PLOT PERFORMANCE METRICS
# ============================================================
plt.figure(figsize=(8, 5))

sns.barplot(
    data=metrics,
    x="Metric",
    y="Score"
)

plt.title("Logistic Regression Performance Metrics")
plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# ============================================================
# 18. FINAL MESSAGE
# ============================================================
print("\n========================================")
print("LOAN PREDICTION COMPLETED SUCCESSFULLY")
print("========================================")