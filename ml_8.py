# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow version:", tf.__version__)
print("All libraries imported successfully!")


# 2. Load the Breast Cancer dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("\nDataset: Breast Cancer Wisconsin")
print("Number of samples:", X.shape[0])
print("Number of input features:", X.shape[1])
print("Class names:", data.target_names)


# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data standardization completed!")


# 5. Define the ANN model
def build_ann(hidden_activation="relu"):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(16, activation=hidden_activation),
        Dense(8, activation=hidden_activation),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


print("\nANN architecture defined successfully!")


# 6. Define the model evaluation function
def evaluate_model(model, experiment, epochs, batch_size):
    probabilities = model.predict(
        X_test,
        verbose=0
    ).ravel()

    y_pred = (probabilities >= 0.5).astype(int)

    test_loss, _ = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    return {
        "Experiment": experiment,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "Test Loss": test_loss
    }


results = []

print("Evaluation function created successfully!")


# 7. Compare different numbers of epochs
epoch_values = [20, 50, 100]

for epochs in epoch_values:
    tf.keras.backend.clear_session()

    model = build_ann("relu")

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    results.append(
        evaluate_model(
            model,
            "Epoch Comparison",
            epochs,
            32
        )
    )

print("\nEpoch comparison completed!")


# 8. Compare different batch sizes
batch_values = [16, 32, 64]

for batch_size in batch_values:
    tf.keras.backend.clear_session()

    model = build_ann("relu")

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=batch_size,
        verbose=0
    )

    results.append(
        evaluate_model(
            model,
            "Batch Size Comparison",
            50,
            batch_size
        )
    )

print("Batch-size comparison completed!")


# 9. Compare different activation functions
activation_values = ["relu", "tanh", "sigmoid"]

for activation in activation_values:
    tf.keras.backend.clear_session()

    model = build_ann(activation)

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    results.append(
        evaluate_model(
            model,
            f"Activation: {activation}",
            50,
            32
        )
    )

print("Activation-function comparison completed!")


# 10. Display and save the comparison results
results_df = pd.DataFrame(results)

print("\nANN Performance Comparison:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "ANN_Performance_Comparison.csv",
    index=False
)

print("\nComparison results saved to ANN_Performance_Comparison.csv")


# 11. Plot accuracy comparison
labels = [
    f"{row['Experiment']}\nE={row['Epochs']}, B={row['Batch Size']}"
    for _, row in results_df.iterrows()
]

plt.figure(figsize=(13, 6))

plt.bar(
    range(len(results_df)),
    results_df["Accuracy"]
)

plt.xticks(
    range(len(results_df)),
    labels,
    rotation=45,
    ha="right"
)

plt.ylabel("Test Accuracy")
plt.title("ANN Performance Comparison - Accuracy")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# 12. Plot test loss comparison
plt.figure(figsize=(12, 6))

plt.bar(
    range(len(results_df)),
    results_df["Test Loss"]
)

plt.xticks(
    range(len(results_df)),
    labels,
    rotation=45,
    ha="right"
)

plt.ylabel("Test Loss")
plt.title("ANN Performance Comparison - Test Loss")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# 13. Train the final ANN model
tf.keras.backend.clear_session()

final_model = build_ann("relu")

# Stop training if validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = final_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stopping],
    verbose=0
)

print("\nFinal ANN model trained successfully!")


# 14. Evaluate the final model
test_loss, test_accuracy = final_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

y_probability = final_model.predict(
    X_test,
    verbose=0
).ravel()

y_pred = (y_probability >= 0.5).astype(int)

print("\nFinal ANN Model Results")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
        zero_division=0
    )
)


# 15. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")
plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], data.target_names, rotation=20)
plt.yticks([0, 1], data.target_names)
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# 16. Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 17. Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 18. Display the ANN architecture
print("\nANN Model Summary:")
final_model.summary()

print("\nExperiment completed successfully!")