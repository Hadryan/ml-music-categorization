from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import silhouette_score, euclidean_distances
import numpy as np

from constants import PCA_COMPONENTS, RANDOM_SEED, RFE_FEATURES, K_MIN, K_MAX, K, IRRELEVANT_FEATURES


class TrackClusterer:
    def __init__(self, data):
        self.data = data
        self.feature_names = self._get_features()
        self.X_processed = None
        self.X_reduced = None
        self.cluster_labels = None
        self.cluster_order = None
        self.track_order = None

    def _get_features(self):
        return [col for col in self.data.columns if col not in IRRELEVANT_FEATURES]

    def apply_pca(self):
        print("\nApplying PCA for dimensionality reduction...")
        pca = PCA(n_components=PCA_COMPONENTS)
        self.X_reduced = pca.fit_transform(self.X_processed)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        print(f"Explained variance by each principal component: {explained_variance}")
        print(
            f"Total explained variance by {PCA_COMPONENTS} components: "
            f"{explained_variance.sum():.2f}"
        )

        return cumulative_variance

    def apply_rfe(self):
        print("\nPerforming Initial Clustering to Obtain Labels for RFE...")
        kmeans_initial = KMeans(n_clusters=K, init='k-means++', random_state=RANDOM_SEED)
        cluster_labels_initial = kmeans_initial.fit_predict(self.X_processed)

        print("\nApplying RFE...")
        model = RandomForestClassifier(random_state=RANDOM_SEED)
        rfe = RFE(estimator=model, n_features_to_select=RFE_FEATURES)
        rfe.fit(self.X_processed, cluster_labels_initial)

        feature_importances = rfe.estimator_.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]
        importance = feature_importances[sorted_idx]

        feat_names = self.feature_names if self.feature_names else [f"Feature {i}"
                                                                    for i in range(self.X_processed.shape[1])]
        feature_names = [feat_names[i] for i in sorted_idx]
        feature_mask = rfe.support_

        self.X_reduced = self.X_processed[:, feature_mask]

        print(f"Number of features reduced to {RFE_FEATURES} using RFE.")
        print(f"Selected feature indices: {np.where(feature_mask)[0]}")

        return feature_names, importance

    def calculate_elbow_scores(self):
        print("\nCalculating Elbow scores (WCSS)...")
        wcss = []
        for k in range(K_MIN, K_MAX + 1):
            print(f"Running K-Means for k = {k}...")
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED)
            kmeans.fit(self.X_reduced)
            wcss.append(kmeans.inertia_)
            print(f"WCSS for k = {k}: {kmeans.inertia_}")
        print("Elbow Method scores calculated successfully.")
        return wcss

    def calculate_silhouette_scores(self):
        print("\nCalculating Silhouette scores...")
        sil_scores = []
        for k in range(K_MIN, K_MAX + 1):
            print(f"Running K-Means for k = {k} to calculate Silhouette score...")
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED)
            labels = kmeans.fit_predict(self.X_reduced)
            score = silhouette_score(self.X_reduced, labels)
            sil_scores.append(score)
            print(f"Silhouette score for k = {k}: {score}")
        print("Silhouette scores calculated successfully.")
        return sil_scores

    def perform_clustering(self):
        print("\nClustering...")
        print(f"Using k = {K} for clustering.")
        kmeans = KMeans(n_clusters=K, random_state=RANDOM_SEED)
        self.cluster_labels = kmeans.fit_predict(self.X_reduced)
        self.data['cluster'] = self.cluster_labels
        return kmeans

    def sort_clusters_by_centroid_distance(self, cluster_centers):
        print("\nSorting clusters based on centroid proximity...")
        num_clusters = cluster_centers.shape[0]
        dist_matrix = euclidean_distances(cluster_centers)
        visited = np.zeros(num_clusters, dtype=bool)
        order = []
        current = 0
        visited[current] = True
        order.append(current)

        while len(order) < num_clusters:
            dists = dist_matrix[current]
            dists[visited] = np.inf
            nearest = np.argmin(dists)
            order.append(nearest)
            visited[nearest] = True
            current = nearest
        print(f"Cluster ordering by centroid distance: {order}")
        self.cluster_order = order

    def sort_tracks_within_cluster(self, X_scaled, track_indices):
        pairwise_dist = euclidean_distances(X_scaled[track_indices])
        visited = np.zeros(len(track_indices), dtype=bool)
        ordering = []
        current = 0
        visited[current] = True
        ordering.append(track_indices[current])

        while len(ordering) < len(track_indices):
            dists = pairwise_dist[current]
            dists[visited] = np.inf
            nearest = np.argmin(dists)
            ordering.append(track_indices[nearest])
            visited[nearest] = True
            current = nearest
        return ordering

    def sort_tracks_by_cluster_order(self, cluster_centers):
        self.sort_clusters_by_centroid_distance(cluster_centers)
        track_order = []
        for cluster_idx in self.cluster_order:
            track_indices = np.where(self.data['cluster'] == cluster_idx)[0]
            sorted_indices = self.sort_tracks_within_cluster(self.X_reduced, track_indices)
            track_order.extend(sorted_indices)
        self.track_order = track_order

    def calculate_nearest_neighbor_distance(self):
        nn_distance = np.zeros(len(self.data), dtype=float)
        for i in range(len(self.track_order) - 1):
            current_idx = self.track_order[i]
            next_idx = self.track_order[i + 1]
            dist = np.linalg.norm(self.X_reduced[current_idx] - self.X_reduced[next_idx])
            nn_distance[current_idx] = dist
        nn_distance[self.track_order[-1]] = np.nan
        return nn_distance

    def save_results(self, output_file='clustered_playlist.csv'):
        track_index = np.zeros(len(self.data), dtype=int)
        for rank, track_id in enumerate(self.track_order):
            track_index[track_id] = rank
        self.data['track_index'] = track_index
        self.data['nn_distance'] = self.calculate_nearest_neighbor_distance()

        df_sorted = self.data.sort_values(by='track_index')
        df_sorted[['artist', 'title', 'genre', 'cluster', 'track_index', 'nn_distance']].to_csv(
            output_file, index=False
        )
        print(f"\nFinal results saved to '{output_file}'.")
        print(
            "Clusters form blocks, sorted by centroid similarity. "
            "Tracks within each cluster are sorted by nearest neighbor. "
            "The 'nn_distance' column shows how far each track is from its next track in the final ordering."
        )
