# General Settings
RANDOM_SEED = 42

# Data Cleaning and Preprocessing
VARIANCE_THRESHOLD = 0.01  # Minimum variance for feature selection
CORRELATION_THRESHOLD = 0.9  # Maximum correlation for feature removal
# TODO - Target feature should be here
IRRELEVANT_FEATURES = ['title', 'artist', 'genre']  # Features to be dropped

# Train-Test Splitting
TRAIN_TEST_SPLIT_RATIO = 0.2  # Proportion of data for testing

# Feature Selection
FEATURE_SELECTION_TOP_N = 80  # Number of features to keep
CLASS_MIN_SAMPLES = 20  # Minimum number of samples required per class

# columns to show in analysis and plots
ANALYSIS_COLUMNS = [
    'tempo', 'rms_mean', 'dynamic_range_db',
    'spectral_centroid_mean', 'spectral_bandwidth_mean',
    'spectral_flatness_mean', 'spectral_rolloff_mean',
    'zcr_mean', 'mfcc_1_mean',
    'chroma_A_mean', 'chroma_C_mean', 'chroma_D_mean',
    'chroma_G_mean', 'tonnetz_1_mean',
    'tempogram_ratio_factor1_mean'
]

# Mapping to group genres
GENRE_MAP = {
    'bass_house_': 'Bass House',
    'chill_': 'Chill',
    'classic_': 'Classic',
    'deep_house_': 'Deep House',
    'drum_n_bass': 'Drum & Bass',
    'edm': 'EDM',
    'electronica': 'Electronica',
    'garage': 'Garage',
    'groovy_': 'Groovy',
    'house_': 'House',
    'tech_house_': 'Tech House',
    'techno_': 'Techno',
    'trance': 'Trance',
    'trap': 'Trap',
    # Vibe genres
    '_bh_': 'Bass House',
    '_dh_': 'Deep House',
    '_e_': 'Electronica',
    '_g_': 'Garage',
    '_h_': 'House',
    '_th_': 'Tech House',
}

# Grid Search settings
GRID_SEARCH_CV = 5  # Number of cross-validation folds
GRID_SEARCH_N_JOBS = -1  # Number of parallel jobs (-1 uses all available cores)

# Grid Search Hyperparameters, will evaluate with all in list
N_ESTIMATORS = [50, 100, 150, 200]  # Number of trees in Random Forest
MAX_DEPTH = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]  # Maximum depth of each tree
MIN_SAMPLES_SPLIT = [2, 5]  # Minimum samples to split a node
SCORING = 'accuracy'  # Metric used for evaluation during Grid Search

# Combine all parameters into a Grid Search Dictionary
PARAM_GRID = {
    'n_estimators': N_ESTIMATORS,
    'max_depth': MAX_DEPTH,
    'min_samples_split': MIN_SAMPLES_SPLIT
}


# Random Forest Default Hyperparameters
RANDOM_FOREST_N_ESTIMATORS = 100  # Default number of trees
RANDOM_FOREST_MAX_DEPTH = 25  # Maximum tree depth
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 2  # Minimum samples to split a node


# KMeans Clustering Parameters
K_MIN = 10  # Minimum number of clusters for KMeans
K_MAX = 100  # Maximum number of clusters for KMeans
K = 100  # Number of clusters to use for KMeans

# Dimensionality Reduction
USE_PCA = True  # Whether to apply PCA or RFE
PCA_COMPONENTS = 70  # Number of principal components

# Recursive Feature Elimination
RFE_FEATURES = 60  # Number of features to retain via RFE



