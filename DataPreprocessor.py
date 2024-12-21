import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

from constants import (
    RANDOM_SEED, TRAIN_TEST_SPLIT_RATIO, CLASS_MIN_SAMPLES,
    FEATURE_SELECTION_TOP_N, RANDOM_FOREST_N_ESTIMATORS, GENRE_MAP
)


class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        # Immediately select only numeric columns
        self.data = data.copy()

        self.features = None
        self.target = None
        self.label_encoder = None

        # For supervised usage:
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_bal = None
        self.y_train_bal = None

        # For unsupervised usage:
        self.X = None

    def drop_columns(self, columns_to_drop):
        print(f"\nDropping features: {columns_to_drop}")
        self.data.drop(columns=columns_to_drop, inplace=True)

    def filter_rare_classes(self, target_col='genre', min_samples=CLASS_MIN_SAMPLES):
        """
        For supervised: Remove classes with fewer than min_samples.
        """
        class_counts = self.data[target_col].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        self.data = self.data[self.data[target_col].isin(valid_classes)]

    def map_genres(self):
        """
        Maps the genres in the target column using the genre_map.
        """
        print(f"Applying genre mapping with {len(GENRE_MAP)} mappings.")
        self.data['genre'] = self.data['genre'].apply(
            lambda x: next((GENRE_MAP[key] for key in GENRE_MAP if key in str(x).lower()), x)
        )


    def encode_label(self, target_col='genre'):
        """
        For supervised: label-encode the target.
        """
        self.label_encoder = LabelEncoder()
        self.data[target_col] = self.label_encoder.fit_transform(self.data[target_col])

    def split_features_and_target(self, target_col='genre'):
        self.features = self.data.drop(columns=[target_col])
        self.target = self.data[target_col]

    def create_train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.target,
            test_size=TRAIN_TEST_SPLIT_RATIO,
            random_state=RANDOM_SEED,
            stratify=self.target
        )

    def balance_with_smote(self):
        smote = SMOTE(random_state=RANDOM_SEED)
        self.X_train_bal, self.y_train_bal = smote.fit_resample(
            self.X_train.values, self.y_train
        )

    def select_top_features(self, top_n=FEATURE_SELECTION_TOP_N):
        """
        Trains a temporary RF on the balanced data, sorts importances,
        and keeps only the top_n features in X_train_bal, X_test.
        """
        temp_rf = RandomForestClassifier(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            random_state=RANDOM_SEED
        )
        temp_rf.fit(self.X_train_bal, self.y_train_bal)

        importances = temp_rf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        original_cols = self.features.columns  # all columns prior to train_test_split
        self.top_features = original_cols[sorted_indices[:top_n]]

        # Slice the arrays
        self.X_train_bal = self.X_train_bal[:, sorted_indices[:top_n]]
        self.X_test = self.X_test.values[:, sorted_indices[:top_n]]

    def scale_features(self):
        """
        StandardScaler for numeric data in self.data -> self.X
        """
        scaler = StandardScaler()
        # Our DataFrame is already guaranteed numeric
        self.X = scaler.fit_transform(self.data)

    def variance_threshold(self, threshold=0.01):
        """
        Removes features with variance below the threshold.
        """
        selector = VarianceThreshold(threshold=threshold)
        if self.X is None:
            # If user didn't call scale_features yet:
            self.X = self.data.values
        self.X = selector.fit_transform(self.X)

    def remove_correlated_features(self, high_corr_threshold=0.9):
        """
        Removes highly correlated features by scanning the correlation matrix of self.X.
        """
        if self.X is None:
            raise ValueError("You must have a numeric array in self.X before removing correlated features.")

        temp_df = pd.DataFrame(self.X)
        corr_matrix = temp_df.corr()

        high_corr_features = set()
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
                    high_corr_features.add(j)

        self.X = np.delete(self.X, list(high_corr_features), axis=1)

    def apply_pca(self, pca_components=70):
        """
        Optionally apply PCA to reduce dimensionality to `pca_components`.
        """
        if self.X is None:
            raise ValueError("You must have a numeric array in self.X before applying PCA.")

        pca = PCA(n_components=pca_components)
        self.X = pca.fit_transform(self.X)

    def get_processed(self):
        if self.X is not None:
            return self.X
        return self.data.values

