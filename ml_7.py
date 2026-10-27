import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 1. Import libraries
print("All libraries imported successfully!")

# 2. Load the Iris dataset
iris = load_iris()

# Select the first two features
X = pd.DataFrame(
    iris.data[:, [0, 1]],
    columns=["Sepal Length", "Sepal Width"]
)

print("\nFirst five rows of the dataset:")
print(X.head())

print("\nDataset shape:", X.shape)

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature standardization completed!")

# 4. Apply DBSCAN clustering
dbscan = DBSCAN(
    eps=0.4,
    min_samples=5
)

cluster_labels = dbscan.fit_predict(X_scaled)

print("DBSCAN clustering completed!")

# 5. Store cluster assignments
result = X.copy()
result["Cluster"] = cluster_labels

print("\nCluster assignments:")
print(result.head(10))

# 6. Analyze the clusters
unique_labels = set(cluster_labels)

clusters = sorted(
    label for label in unique_labels
    if label != -1
)

noise_count = np.sum(cluster_labels == -1)
cluster_count = len(clusters)

print("\nDBSCAN Parameters")
print("eps =", dbscan.eps)
print("min_samples =", dbscan.min_samples)

print("\nNumber of clusters:", cluster_count)
print("Number of noise points:", noise_count)

print("\nPoints in each cluster:")

for label in clusters:
    count = np.sum(cluster_labels == label)
    print(f"Cluster {label}: {count} points")

print(f"Noise (-1): {noise_count} points")

# 7. Identify core points
core_indices = dbscan.core_sample_indices_

print("\nNumber of core points:", len(core_indices))

# 8. Visualize the clusters
plt.figure(figsize=(9, 6))

for label in sorted(unique_labels):
    mask = cluster_labels == label

    if label == -1:
        plt.scatter(
            X.loc[mask, "Sepal Length"],
            X.loc[mask, "Sepal Width"],
            c="black",
            marker="x",
            s=80,
            label="Noise"
        )
    else:
        plt.scatter(
            X.loc[mask, "Sepal Length"],
            X.loc[mask, "Sepal Width"],
            s=50,
            label=f"Cluster {label}"
        )

# Highlight core points with red outlines
if len(core_indices) > 0:
    plt.scatter(
        X.iloc[core_indices, 0],
        X.iloc[core_indices, 1],
        s=110,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="Core Points"
    )

plt.title("DBSCAN Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. Calculate the silhouette score, excluding noise
non_noise_mask = cluster_labels != -1
non_noise_labels = cluster_labels[non_noise_mask]
non_noise_data = X_scaled[non_noise_mask]

if (
    len(set(non_noise_labels)) >= 2
    and len(non_noise_labels) > len(set(non_noise_labels))
):
    score = silhouette_score(
        non_noise_data,
        non_noise_labels
    )

    print(
        "\nSilhouette Score (excluding noise):",
        round(score, 4)
    )
else:
    print(
        "\nSilhouette Score not calculated: "
        "at least two non-noise clusters are required."
    )

# 10. Display noise points
noise_data = result[result["Cluster"] == -1]

print("\nNoise Points:")
print(noise_data)

# 11. Save the results to a CSV file
result.to_csv(
    "DBSCAN_Clustering_Results.csv",
    index=False
)

print("\nResults saved to DBSCAN_Clustering_Results.csv")