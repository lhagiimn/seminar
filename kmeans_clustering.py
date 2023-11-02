import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering


iris = load_iris()
data = iris.data[:, [1, 3]]

scaler = MinMaxScaler()
scaler = scaler.fit(data)
scaled_data = scaler.transform(data)

labels = KMeans(n_clusters=3, random_state=42).fit_predict(scaled_data)

agg_cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean").fit(scaled_data)

children = agg_cluster.children_ # 1
distance = np.arange(children.shape[0]) # 2
no_of_observations = np.arange(2, children.shape[0]+2)

linkage_matrix = np.hstack((children, distance[:, np.newaxis],
                            no_of_observations[:, np.newaxis])).astype(float)

fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
dendrogram(linkage_matrix, leaf_font_size=8)
plt.show()

fig, ax = plt.subplots()
# group 1
ax.scatter(data[labels == 0, 0], data[labels == 0, 1], s=30, marker='o', label='group 1')
# group 2
ax.scatter(data[labels == 1, 0], data[labels == 1, 1], s=30, marker='v', label='group 2')
# group 3
ax.scatter(data[labels == 2, 0], data[labels == 2, 1], s=30, marker='s', label='group 3')

ax.set_xlabel('sepal width (cm)')
ax.set_ylabel('petal width (cm)')
ax.legend(loc='best')
plt.show()