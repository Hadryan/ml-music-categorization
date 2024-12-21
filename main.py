import pandas as pd

from DataPreprocessor import DataPreprocessor
from GenreClassifier import GenreClassifier
from MusicDataAnalyzer import MusicDataAnalyzer
from TrackClusterer import TrackClusterer
from constants import CLASS_MIN_SAMPLES, GRID_SEARCH_CV, GRID_SEARCH_N_JOBS, K_MIN, K_MAX, USE_PCA, \
    GENRE_MAP, IRRELEVANT_FEATURES


def run_supervised(df):
    print("\n--- Supervised Pipeline ---")

    # Preprocess data
    preprocessor = DataPreprocessor(df)
    preprocessor.drop_columns(['artist', 'title'])
    preprocessor.map_genres()

    # Filter underrepresented genres by a certain threshold
    preprocessor.filter_rare_classes(target_col='genre', min_samples=CLASS_MIN_SAMPLES)

    # Encode the 'genre' column
    preprocessor.encode_label(target_col='genre')

    # Split features and target
    preprocessor.split_features_and_target(target_col='genre')

    # Train-test split
    preprocessor.create_train_test_split()

    # Create samples for underrepresented classes using SMOTE
    preprocessor.balance_with_smote()

    # Select top features based on a random forrest classifier
    preprocessor.select_top_features()

    # Instantiate your classifier with processed data
    classifier = GenreClassifier(
        X_train_bal=preprocessor.X_train_bal,
        y_train_bal=preprocessor.y_train_bal,
        X_test=preprocessor.X_test,
        y_test=preprocessor.y_test,
        label_encoder=preprocessor.label_encoder,
        top_features=preprocessor.top_features
    )

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    classifier.tune_model_with_grid_search(
        param_grid=param_grid,
        cv=GRID_SEARCH_CV,
        scoring='accuracy',
        n_jobs=GRID_SEARCH_N_JOBS
    )

    # Evaluate model
    classifier.evaluate_model()
    classifier.plot_feature_importance()

    # Plot results
    classifier.plot_pairwise_relationships()
    classifier.plot_precision_recall_curve()

    # Return the classifier in case we want to export the trained model
    return classifier


def run_unsupervised(df, music_data_analyzer):
    preprocessor = DataPreprocessor(df)
    clusterer = TrackClusterer(df)

    # Preprocess data
    preprocessor.drop_columns(IRRELEVANT_FEATURES)

    preprocessor.scale_features()  # Scale (normalize) features
    preprocessor.variance_threshold(threshold=0.01)  # Apply Variance Threshold
    preprocessor.remove_correlated_features(high_corr_threshold=0.9)  # Remove correlated features

    # Retrieve processed data
    X_processed = preprocessor.get_processed()
    print(f"Shape of X_processed: {X_processed.shape}")

    # Pass processed data to the clusterer
    clusterer.X_processed = X_processed

    # Apply Dimensionality reduction / feature selection
    if USE_PCA:
        cumulative_variance = clusterer.apply_pca()
        music_data_analyzer.plot_pca_variance(cumulative_variance)
    else:
        feature_names, importance = clusterer.apply_rfe()
        music_data_analyzer.plot_feature_importances(feature_names, importance)

    # Calculate clustering scores
    wcss = clusterer.calculate_elbow_scores()
    sil_scores = clusterer.calculate_silhouette_scores()

    # Perform clustering
    kmeans_model = clusterer.perform_clustering()

    # Sort clusters and save results to csv
    clusterer.sort_tracks_by_cluster_order(kmeans_model.cluster_centers_)
    clusterer.save_results('clustered_playlist.csv')

    k_range = range(K_MIN, K_MAX + 1)
    music_data_analyzer.plot_elbow(k_range, wcss)
    music_data_analyzer.plot_silhouette(k_range, sil_scores)
    music_data_analyzer.visualize_clusters_pca(clusterer.X_reduced, clusterer.cluster_labels)


def main():
    print("Loading data...")
    df = pd.read_csv("data/track_features.csv")

    # Explore data
    music_data_analyzer = MusicDataAnalyzer(df)
    music_data_analyzer.extract_chord_patterns()
    music_data_analyzer.display_class_distribution(target="genre")
    music_data_analyzer.display_class_distribution("genre", figsize=(10, 8), genre_map=GENRE_MAP)
    music_data_analyzer.display_statistics()
    music_data_analyzer.display_correlations()
    music_data_analyzer.plot_feature_histograms()
    music_data_analyzer.display_box_plots()
    music_data_analyzer.display_outliers()

    # 1: Run supervised model
    run_supervised(df)

    # 2: Run unsupervised model
    # run_unsupervised(df, music_data_analyzer)

if __name__ == '__main__':
    main()
