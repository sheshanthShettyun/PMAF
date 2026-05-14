# Converted from PMAF_6/Loan.ipynb
# Original notebook code cells exported as a normal Python script.

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("loan.csv")

# First 5 rows
df.head()

# Drop non-numeric / irrelevant columns
df_clean = df.select_dtypes(include=[np.number])

# Handle missing values
df_clean = df_clean.fillna(df_clean.mean())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Dendrogram
plt.figure(figsize=(10, 6))
linked = linkage(X_scaled, method='ward')

dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Apply Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X_scaled)

# Add cluster labels to original dataset
df["Cluster"] = labels

# First 5 rows
df.head()

# Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='rainbow')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Hierarchical Clustering (Loan Dataset)")
plt.show()
