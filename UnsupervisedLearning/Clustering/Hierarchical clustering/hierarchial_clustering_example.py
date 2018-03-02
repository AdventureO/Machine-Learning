import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

# Create data (points in two dimesional plane)
x = np.array([[1, 1], [1.5, 1], [3, 3], [4, 4], [3, 3.5], [3.5, 4]])

# Show our points on the plane
plt.scatter(x[:, 0], x[:, 1], s=50)
plt.show()

# Create linkage matrix(distance between points) for building dendrogram using nearest point algorithm
# d(u, v) = min(dist(point_i, point_j))
linkage_matrix = linkage(x, "single")

# Create dendrogram
dendrogram = dendrogram(linkage_matrix, truncate_mode='none')

# Show dendrogram
plt.title("Hierarchical clustering")
plt.show()