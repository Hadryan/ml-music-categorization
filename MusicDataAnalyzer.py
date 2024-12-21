import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from constants import ANALYSIS_COLUMNS, RFE_FEATURES


class MusicDataAnalyzer:
    def __init__(self, data):
        self.data = data

    def display_shape(self):
        print(f"Number of data points (rows): {self.data.shape[0]}")
        print(f"Number of features (columns): {self.data.shape[1]}")

    def display_missing_values(self):
        total_missing = self.data.isnull().sum().sum()
        print(f"Total missing values in the dataset: {total_missing}")

        if total_missing > 0:
            missing_by_col = self.data.isnull().sum().sort_values(ascending=False)
            print("Missing values by column:")
            print(missing_by_col[missing_by_col > 0])

    def display_outliers(self):
        numeric_cols = self.data.select_dtypes(include='number').columns
        total_rows = len(self.data)

        print("Outlier Summary (Number of Outliers per Column):")
        print(f"Total rows in dataset: {total_rows}")
        print("-" * 50)

        outlier_found = False

        for col in numeric_cols:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            num_outliers = len(outliers)

            if num_outliers > 0:
                percentage = (num_outliers / total_rows) * 100
                print(f"{col:<30} {num_outliers:>5} data points ({percentage:>5.2f}%)")
                outlier_found = True

        if not outlier_found:
            print("No outliers detected in numeric columns.")

        print("-" * 50)

    def display_class_distribution(self, target, figsize=(25, 8), genre_map=None):
        if genre_map:
            grouped_genres = self.data[target].apply(
                lambda x: next((genre_map[key] for key in genre_map if key in str(x).lower()), x)
            )
            distribution = grouped_genres.value_counts()
            title = f"Distribution of {target} (Grouped)"
        else:
            distribution = self.data[target].value_counts()
            title = f"Distribution of {target}"

        print(f"Distribution for '{target}':")
        print(distribution)

        plt.figure(figsize=figsize)
        distribution.plot(kind='bar', title=title)
        plt.xlabel(target)
        plt.ylabel("Count")
        plt.xticks(fontsize=10, rotation=45)
        plt.tight_layout()
        plt.show()

    def display_statistics(self):
        with pd.option_context('display.max_columns', None):
            print(self.data[ANALYSIS_COLUMNS].describe())

    def display_correlations(self, method='pearson'):
        numeric_data = self.data.select_dtypes(include=['number'])
        corr = numeric_data.corr(method=method)
        print(f"Correlation matrix using {method} method:\n", corr)

        plt.figure(figsize=(20, 16), dpi=100)
        sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm')
        plt.title(f"Correlation Heatmap ({method.capitalize()})")
        plt.tight_layout()
        plt.show()

    def extract_chord_patterns(self):
        chroma_columns = [col for col in self.data.columns if col.startswith('chroma_')]
        chroma_features = self.data[chroma_columns]
        correlation_matrix = chroma_features.corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=True,
            yticklabels=True,
            annot_kws={"size": 8},
        )
        plt.title("Chroma Feature Correlation (Chord Relationships)", fontsize=16)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_feature_histograms(self):
        valid_cols = [col for col in ANALYSIS_COLUMNS if col in self.data.columns]
        if not valid_cols:
            print("No valid columns found for plotting.")
            return

        self.data[valid_cols].hist(bins=30, figsize=(10, 8))
        plt.tight_layout()
        plt.show()

    def display_box_plots(self):
        valid_cols = [col for col in ANALYSIS_COLUMNS if col in self.data.columns]
        if not valid_cols:
            print("No valid columns found for plotting.")
            return

        plt.figure(figsize=(10, 8), dpi=100)
        self.data[valid_cols].boxplot()
        plt.title("Box plots of Selected Features")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_elbow(self, k_range, wcss):
        """
        Plot the Elbow Method results, given a range of k and their WCSS values.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, wcss, marker='o')
        plt.title('Elbow Method (WCSS vs K)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.show()

    def plot_silhouette(self, k_range, sil_scores):
        """
        Plot the Silhouette scores, given a range of k and their Silhouette values.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, sil_scores, marker='o', color='orange')
        plt.title('Silhouette Scores vs Number of Clusters')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.show()

    def plot_pca_variance(self, cumulative_variance):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.axhline(y=0.8, color='r', linestyle='--', label="80% variance")
        plt.axhline(y=0.95, color='g', linestyle='--', label="95% variance")
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Explained Variance")
        plt.legend()
        plt.show()

    def visualize_clusters_pca(self, X_reduced, cluster_labels):
        pca_2d = PCA(n_components=2)
        X_pca = pca_2d.fit_transform(X_reduced)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=10)
        plt.title("Clusters Visualized with PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster")
        plt.show()

    def plot_feature_importances(self, feature_names, importance):
        plt.figure(figsize=(10, 6))
        plt.barh(
            feature_names[:RFE_FEATURES][::-1],
            importance[:RFE_FEATURES][::-1],
            color='skyblue'
        )
        plt.xlabel("Feature Importance")
        plt.title("Top Features Selected by RFE (Random Forest)")
        plt.show()