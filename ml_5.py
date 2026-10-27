# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("All libraries imported successfully!")


# ============================================================
# 2. LOAD THE IRIS DATASET
# ============================================================
iris = load_iris()

X = iris.data
y = iris.target
class_names = iris.target_names

print("\nDataset: Iris")
print("Total Samples:", X.shape[0])
print("Number of Features:", X.shape[1])
print("Features:", iris.feature_names)
print("Classes:", list(class_names))


# ============================================================
# 3. SPLIT THE DATASET
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
# 4. CREATE SVM MODELS
# ============================================================
models = {
    "Linear SVM": SVC(
        kernel="linear",
        C=1.0
    ),

    "RBF SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )
}

results = []
confusion_matrices = {}

print("\nLinear and RBF SVM models created successfully!")


# ============================================================
# 5. TRAIN AND EVALUATE BOTH MODELS
# ============================================================
for model_name, classifier in models.items():

    # Create a pipeline with feature scaling and SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", classifier)
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predict test labels
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1-Score": f1
    })

    # Calculate confusion matrix
    confusion_matrices[model_name] = confusion_matrix(
        y_test,
        y_pred
    )

    # Display evaluation results
    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-Score  : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )


# ============================================================
# 6. COMPARE MODEL PERFORMANCE
# ============================================================
results_df = pd.DataFrame(results).set_index("Model")

print("\nPerformance Comparison:")

print(results_df.round(4).to_string())


# ============================================================
# 7. VISUALIZE CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

for ax, (model_name, cm) in zip(
    axes,
    confusion_matrices.items()
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.suptitle("Confusion Matrix Comparison: SVM Models")
plt.tight_layout()
plt.show()


# ============================================================
# 8. VISUALIZE PERFORMANCE METRICS
# ============================================================
results_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Performance Comparison of SVM Models")
plt.xlabel("SVM Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 9. IDENTIFY THE BEST MODEL
# ============================================================
best_model = results_df["Macro F1-Score"].idxmax()

print("\n" + "=" * 55)
print("BEST MODEL")
print("=" * 55)

print(
    "Model with the highest test macro F1-score:",
    best_model
)

print(
    "Highest Macro F1-Score:",
    round(results_df.loc[best_model, "Macro F1-Score"], 4)
)

print(
    "\nNote: This comparison uses one stratified train-test split."
)

print("\nIris Flower Classification Completed Successfully!")