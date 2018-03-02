import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# Generate data(random coordinates)
X,Y = make_blobs(n_samples=100, centers=5, random_state=0, cluster_std=0.8) #standart deviation

# Show row data
plt.scatter(X[:,0], X[:,1], s=50)
plt.show()

# Create k-means estimator
estimator = KMeans(n_clusters=5)
# Train estimator
estimator.fit(X)
# Get the prediction from estimator
y_kmeans = estimator.predict(X)

# Show clustered data
plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='rainbow')
plt.show()