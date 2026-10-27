# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("All libraries imported successfully!")


# ============================================================
# 2. LOAD THE DATASET
# ============================================================
df = pd.read_csv(
    r"C:\Users\staff\Downloads\archive\Mall_Customers.csv"
)

print("\nFirst Five Rows of the Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 3. SELECT FEATURES FOR CUSTOMER SEGMENTATION
# ============================================================
features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

if not all(col in df.columns for col in features):
    raise ValueError(
        "Required columns not found. Check the CSV column names."
    )

X = df[features].apply(
    pd.to_numeric,
    errors="coerce"
)

# Remove rows containing missing or invalid feature values
X = X.dropna()

print("\nNumber of Records Used:", len(X))

print("\nSelected Features:")
print(X.head())


# ============================================================
# 4. STANDARDIZE THE FEATURES
# ============================================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Standardization Completed!")
print("Scaled Data Shape:", X_scaled.shape)


# ============================================================
# 5. FIND OPTIMAL K USING THE ELBOW METHOD
# ============================================================
wcss = []
k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

    print(f"K = {k}, WCSS = {kmeans.inertia_:.2f}")


# Plot the Elbow Curve
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    wcss,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 6. EVALUATE CLUSTERS USING SILHOUETTE SCORE
# ============================================================
silhouette_scores = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.4f}"
    )


# Plot Silhouette Scores
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 7. SELECT OPTIMAL K
# ============================================================
best_index = np.argmax(silhouette_scores)

optimal_k = list(k_values)[best_index]

best_score = silhouette_scores[best_index]

print("\nOptimal K Based on Silhouette Score:", optimal_k)

print(
    "Best Silhouette Score:",
    round(best_score, 4)
)


# ============================================================
# 8. TRAIN THE FINAL K-MEANS MODEL
# ============================================================
final_model = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

cluster_labels = final_model.fit_predict(X_scaled)

print("\nFinal K-Means Model Trained Successfully!")


# ============================================================
# 9. ADD CLUSTER LABELS TO THE DATASET
# ============================================================
result = X.copy()

result["Cluster"] = cluster_labels

print("\nCluster Assignments:")
print(result.head(10))

print("\nNumber of Customers in Each Cluster:")
print(
    result["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. CONVERT CLUSTER CENTROIDS TO ORIGINAL SCALE
# ============================================================
centroids_scaled = final_model.cluster_centers_

centroids_original = scaler.inverse_transform(
    centroids_scaled
)


# ============================================================
# 11. VISUALIZE CUSTOMER SEGMENTS
# ============================================================
plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=50,
    alpha=0.8
)

plt.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    marker="X",
    s=250,
    c="red",
    edgecolors="black",
    label="Centroids"
)

plt.title(
    f"K-Means Customer Segmentation (K={optimal_k})"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.legend()

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()
plt.show()


# ============================================================
# 12. ANALYZE EACH CUSTOMER CLUSTER
# ============================================================
cluster_summary = result.groupby("Cluster")[features].mean().round(2)

print("\nCluster-Wise Feature Averages:")
print(cluster_summary)

print("\nCluster-Wise Customer Counts:")
print(
    result.groupby("Cluster")
    .size()
    .rename("Number of Customers")
)


# ============================================================
# 13. SAVE THE RESULTS TO CSV
# ============================================================
output_path = (
    r"C:\Users\staff\Downloads\archive"
    r"\KMeans_Cluster_Results.csv"
)

result.to_csv(
    output_path,
    index=False
)

print("\n========================================")
print("CUSTOMER SEGMENTATION COMPLETED")
print("========================================")

print("Results Saved Successfully!")

print("File Location:", output_path)