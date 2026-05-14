
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "iris.csv")

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

print("First 5 Rows:")
print(data.head())

X = data.iloc[:, :4]

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X)

clusters = kmeans.predict(X)

data['Cluster'] = clusters

print("\nClustered Data:")
print(data.head())

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering on Iris Dataset")

plt.show()

loan_path = os.path.join(SCRIPT_DIR, "loan.csv")
loan_data = pd.read_csv(loan_path)

print("\nLoan Dataset First 5 Rows:")
print(loan_data.head())

loan_features = pd.get_dummies(loan_data.drop("Loan_Status", axis=1), drop_first=True)
loan_scaled = StandardScaler().fit_transform(loan_features)
linked = linkage(loan_scaled, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram - Loan Dataset")
plt.xlabel("Applicants")
plt.ylabel("Distance")
plt.show()
