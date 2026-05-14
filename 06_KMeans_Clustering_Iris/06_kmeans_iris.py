# Serial mapped from current codebase: Clustering (Unsupervised Learning).py
# Cleaned current-codebase serial file.

# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "iris.csv")

# Load Iris dataset
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

print("First 5 Rows:")
print(data.head())

# Select features
X = data.iloc[:, :4]

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit model
kmeans.fit(X)

# Predicted clusters
clusters = kmeans.predict(X)

# Add cluster column
data['Cluster'] = clusters

print("\nClustered Data:")
print(data.head())

# Plot clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering on Iris Dataset")

plt.show()
