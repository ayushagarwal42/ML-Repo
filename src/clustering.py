# clustering.py

from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score

from src.Clustering.DBSCAN import DBSCAN
from src.Clustering.MeanShiftClustering import MeanShift
from src.Clustering.kmeans import KMeans


def train_clustering_models(X):
    # Example: Train and evaluate clustering models
    models = {
        "K-Means": KMeans(K=2),
        "DBSCAN": DBSCAN(eps=0.2, min_samples=5),
        "Mean Shift": MeanShift(radius=2.5)
    }

    for model_name, model in models.items():
        cluster_labels = model.fit(X)
        silhouette_avg = silhouette_score(X, cluster_labels)

        print(f"Model: {model_name}")
        print(f"Silhouette Score: {silhouette_avg}")
        print("-" * 40)
