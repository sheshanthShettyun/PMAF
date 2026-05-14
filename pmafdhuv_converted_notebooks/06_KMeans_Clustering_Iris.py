# Converted from PMAF_6/Iris.ipynb
# Original notebook code cells exported as a normal Python script.

# Import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Iris.csv")

# First 5 rows
df.head()

# Drop non-feature columns (like Id and Species if present)
X = df.drop(columns=["Id", "Species"], errors='ignore')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Get cluster labels
labels = kmeans.labels_

# Add cluster labels to original dataframe
df["Cluster"] = labels

# First 5 rows
df.head()

# Cluster centers
print("Cluster Centers (scaled):")
print(kmeans.cluster_centers_)

# Visualize using first two features
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-means Clustering (Iris)")
plt.show()
