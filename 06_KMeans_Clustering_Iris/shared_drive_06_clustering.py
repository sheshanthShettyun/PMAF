#pip install pandas matplotlib scikit-learn
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Read dataset
DATA_PATH = Path(__file__).with_name("clustering_dataset.csv")
df = pd.read_csv(DATA_PATH)

# Selecting features (drop target column if present)
X = df.iloc[:, :-1]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-MeANS
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

labels = kmeans.labels_

# Visualization (first two features only)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

# Hierarchical Clustering
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()
