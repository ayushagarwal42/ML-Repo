# kmeans.py

import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, K=5, max_iters=100):
        self.max_iters = max_iters
        self.K = K
        self.inertia = None  # To store the inertia value

        # List of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # The centers (mean vector) for each cluster
        self.centroids = []

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize the centroids
        random_samples_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_samples_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to the closest centroids (create clusters)
            self.clusters = self.create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)

            if self.is_converged(centroids_old, self.centroids):
                break

        # Calculate the inertia value (within-cluster sum of squares)
        self.inertia = self.calculate_inertia()

        # Classify samples as the index of their clusters
        return self.get_cluster_labels(self.clusters)

    def calculate_inertia(self):
        inertia = 0
        for i in range(self.K):
            cluster_samples = self.X[self.clusters[i]]
            centroid = self.centroids[i]
            distances = [euclidean_distance(sample, centroid) for sample in cluster_samples]
            inertia += np.sum(np.square(distances))
        return inertia

    def get_cluster_labels(self, clusters):
        # Each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def create_clusters(self, centroids):
        # Assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def get_centroids(self, clusters):
        # Assign the mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_converged(self, centroids_old, centroids):
        # Distances between old and new centroids for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
