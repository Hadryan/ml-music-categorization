import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, RFE
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = 'data/track_features.csv'
IRRELEVANT_FEATURES = ['artist', 'title', 'genre']

RANDOM_SEED = 42
K_MIN = 10
K_MAX = 100
K = 50

# set this to true to enable Principal Component analysis
# if it is false, the code will use Recursive feature eliminatrion
USE_PCA = True

PCA_COMPONENTS = 70

RFE_FEATURES = 20


def load_data(csv_file: str):
    df = pd.read_csv(csv_file)
    print(f"Data loaded, Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    feature_names = [col for col in df.columns if col not in IRRELEVANT_FEATURES]
    return df, feature_names

def pre_process_data(df):
    print(f"\nDropping irrelevant features: {IRRELEVANT_FEATURES}")
    df_numeric = df.drop(columns=[col for col in IRRELEVANT_FEATURES if col in df.columns])

    # Normalize features
    X_scaled = normalize_features(df_numeric)

    # Apply Variance Threshold
    X_reduced = apply_variance_threshold(X_scaled, threshold=0.01)

    # Analyze and Remove Correlated Features
    X_final = analyze_and_remove_correlated_features(X_reduced, high_corr_threshold=0.9)

    return X_final


def normalize_features(df_numeric):
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)
    print(f"Sample of normalized data:\n{X_scaled[:5]}")
    return X_scaled


def apply_variance_threshold(X_scaled, threshold=0.01):
    print("\nApplying Variance Threshold...")
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X_scaled)
    print(f"Number of features reduced from {X_scaled.shape[1]} to {X_reduced.shape[1]} after Variance Thresholding.")
    return X_reduced


def analyze_and_remove_correlated_features(X_reduced, high_corr_threshold=0.9):
    print("\nAnalyzing Correlation Matrix...")
    correlation_matrix = pd.DataFrame(X_reduced).corr()

    # Show correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Identify highly correlated features
    high_corr_features = set()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
                high_corr_features.add(j)

    print(f"Number of highly correlated features: {len(high_corr_features)}")

    # Drop highly correlated features
    X_final = np.delete(X_reduced, list(high_corr_features), axis=1)
    print(f"Number of features after dropping: {X_final.shape[1]}")
    return X_final


def apply_pca(X_scaled):
    print("\nApplying PCA for dimensionality reduction...")
    pca = PCA(n_components=PCA_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    print(f"Explained variance by each principal component: {explained_variance}")
    print(f"Total explained variance by {PCA_COMPONENTS} components: {explained_variance.sum():.2f}")

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.8, color='r', linestyle='--', label="80% variance")
    plt.axhline(y=0.95, color='g', linestyle='--', label="95% variance")
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance")
    plt.legend()
    plt.show()

    return X_pca


def apply_rfe(X_scaled, cluster_labels, feature_names=None):
    """
    Applies Recursive Feature Elimination (RFE) for feature selection and visualizes feature importance.

    Parameters:
    - X_scaled: Scaled feature matrix.
    - cluster_labels: Cluster labels from initial clustering.
    - feature_names: List of feature names (optional, for labeling the plot).

    Returns:
    - X_rfe: Reduced feature matrix after RFE.
    """
    print("\nApplying Recursive Feature Elimination (RFE)...")
    
    # Fit RFE with a Random Forest model
    model = RandomForestClassifier(random_state=RANDOM_SEED)
    rfe = RFE(estimator=model, n_features_to_select=RFE_FEATURES)
    rfe.fit(X_scaled, cluster_labels)

    # Get the feature importances from the Random Forest model
    feature_importances = rfe.estimator_.feature_importances_

    # Sort feature importances for visualization
    sorted_idx = np.argsort(feature_importances)[::-1]  # Descending order
    sorted_importances = feature_importances[sorted_idx]
    sorted_feature_names = feature_names if feature_names is not None else [f"Feature {i}" for i in range(X_scaled.shape[1])]
    sorted_feature_names = [sorted_feature_names[i] for i in sorted_idx]

    # Visualize feature importance
    print(f"Visualizing feature importance for top {RFE_FEATURES} features...")
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_feature_names[:RFE_FEATURES][::-1], sorted_importances[:RFE_FEATURES][::-1], color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("Top Features Selected by RFE (Random Forest)")
    plt.show()

    # Get the bool mask of selected features
    feature_mask = rfe.support_
    print(f"Number of features reduced to {RFE_FEATURES} using RFE.")
    print(f"Selected feature indices: {np.where(feature_mask)[0]}")

    # Apply the mask to reduce the dataset
    X_rfe = X_scaled[:, feature_mask]

    return X_rfe




def calculate_elbow_scores(X_scaled):

    print("\nCalculating Elbow scores (WCSS)...")
    wcss = [] # measures the sum of squared distances between data points
    for k in range(K_MIN, K_MAX + 1):
        print(f"Running K-Means for k = {k}...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        print(f"WCSS for k = {k}: {kmeans.inertia_}")

    print("Elbow Method scores calculated successfully.")
    return wcss


def calculate_silhouette_scores(X_scaled):
    print("\nCalculating Silhouette scores...")
    sil_scores = []
    for k in range(K_MIN, K_MAX + 1):
        print(f"Running K-Means for k = {k} to calculate Silhouette score...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
        print(f"Silhouette score for k = {k}: {score}")
    print("Silhouette scores calculated successfully.")
    return sil_scores


def plot_elbow(k_range, wcss):
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o')
    plt.title('Elbow Method (WCSS vs K)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()


def plot_silhouette(k_range, sil_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sil_scores, marker='o', color='orange')
    plt.title('Silhouette Scores vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()


def visualize_clusters_pca(X_scaled, cluster_labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=10)
    plt.title("Clusters Visualized with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    plt.show()


def sort_clusters_by_centroid_distance(cluster_centers):
    """
    This function uses the center of each cluster to define the ordering, it keeps similar clusters next to each other.
    """
    print("\nSorting clusters based on centroid proximity...")
    num_clusters = cluster_centers.shape[0] 
    dist_matrix = euclidean_distances(cluster_centers) # distancesrom teh center for each cluster
    visited = np.zeros(num_clusters, dtype=bool)
    cluster_order = []
    current = 0 
    visited[current] = True
    cluster_order.append(current)

    while (len(cluster_order) < num_clusters):
        
        # Get distances from the current cluster to all others
        dists = dist_matrix[current]

        # Ignore already visited clusters
        dists[visited] = np.inf

        # get the nearest unvisited cluster
        nearest = np.argmin(dists)

        # Add the nearest cluster to the order
        cluster_order.append(nearest)    

        # set as visited    
        visited[nearest] = True
        
        # update current
        current = nearest

    print(f"Cluster ordering by centroid distance: {cluster_order}")
    return cluster_order


def sort_tracks_within_cluster(X_scaled, track_indices):
    """
    Sorts tracks inside a cluster based on the nearest-neighbor.
    """
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


def sort_tracks_by_cluster_order(cluster_order, df, X_scaled):
    track_order = []
    for cluster_index in cluster_order:
        track_indices = np.where(df['cluster'] == cluster_index)[0]
        sorted_indices = sort_tracks_within_cluster(X_scaled, track_indices)
        track_order.extend(sorted_indices)
    return track_order


def calculate_nearest_neighbor_distance(df, track_order, X_scaled):
    """
    Calculate the nearest-neighbor distance for each track based on the track order.
    """
    nn_distance = np.zeros(len(df), dtype=float)

    for i in range(len(track_order) - 1):
        current_idx = track_order[i]
        next_idx = track_order[i + 1]
        dist = np.linalg.norm(X_scaled[current_idx] - X_scaled[next_idx])
        nn_distance[current_idx] = dist

    # The last track in the order has no next track, assign np.nan
    nn_distance[track_order[-1]] = np.nan

    return nn_distance


def main():

    print("Step 1: Loading data...")
    df, feature_names = load_data(DATA_FILE)

    print("\nStep 2: Data Preprocessing.")
    X_processed  = pre_process_data(df)

    if USE_PCA:
        print("\nStep 3: Applying PCA...")
        X_reduced = apply_pca(X_processed)

    else:
        print("\nStep 3: Performing Initial Clustering to Obtain Labels for RFE...")
        kmeans_initial = KMeans(n_clusters=K, init='k-means++', random_state=RANDOM_SEED)
        cluster_labels_initial = kmeans_initial.fit_predict(X_processed)
        print("\nApplying RFE...")
        X_reduced = apply_rfe(X_processed, cluster_labels_initial)


    print("\nStep 3: Performing Cluster Evaluation.")
    wcss = calculate_elbow_scores(X_reduced)
    sil_scores = calculate_silhouette_scores(X_reduced)

    # Plot Elbow and Silhouette
    k_range = range(K_MIN, K_MAX + 1)
    plot_elbow(k_range, wcss)
    plot_silhouette(k_range, sil_scores)

    print("\nStep 4: Clustering...")
    print(f"Using k = {K} for clustering.")
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_SEED)

    cluster_labels = kmeans.fit_predict(X_reduced)
    df['cluster'] = cluster_labels

    # Visualize clusters using PCA
    print("\nVisualizing clusters with PCA...")
    visualize_clusters_pca(X_reduced, cluster_labels)

    #Sort clusters by centroid distance
    cluster_order = sort_clusters_by_centroid_distance(kmeans.cluster_centers_)

    # Sort tracks within each cluster
    track_order = sort_tracks_by_cluster_order(cluster_order, df, X_reduced)

    # Set the track_index based on track_order
    track_index = np.zeros(len(df), dtype=int) 
    for rank, track_id in enumerate(track_order):
        track_index[track_id] = rank 

    df['track_index'] = track_index

    # Set the nearest neighbout distance for each track
    df['nn_distance'] = calculate_nearest_neighbor_distance(df, track_order, X_reduced)

    # Create sorted DataFrame by track_index
    df_sorted = df.sort_values(by='track_index')

    # Save the CSV with relevant columns
    output_file = 'clustered_playlist.csv'
    df_sorted[['artist', 'title', 'genre', 'cluster', 'track_index', 'nn_distance']].to_csv(output_file, index=False)

    print(f"\nFinal results saved to '{output_file}'.")
    print("Clusters form blocks, and cluster blocks are sorted by centroid similarity. Each track is also sorted within its cluster by nearest neighbor.")
    print("The 'nn_distance' column shows how far each track is from its next track in the final ordering.")

if __name__ == '__main__':
    main()
