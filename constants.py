RANDOM_SEED = 42

# Random Forest hyperparams
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 15
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 2

TRAIN_TEST_SPLIT_RATIO = 0.2
FEATURE_SELECTION_TOP_N = 50
CLASS_MIN_SAMPLES = 40

GRID_SEARCH_CV = 3
GRID_SEARCH_N_JOBS = -1

# KMeans hyperparams
K_MIN = 10
K_MAX = 100
K = 50

USE_PCA = True
PCA_COMPONENTS = 70
RFE_FEATURES = 20

IRRELEVANT_FEATURES = ['title', 'artist', 'genre']

ANALYSIS_COLUMNS = [
    'tempo', 'rms_mean', 'dynamic_range_db',
    'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_flatness_mean', 'spectral_rolloff_mean',
    'zcr_mean',
    'mfcc_1_mean',
    'chroma_A_mean', 'chroma_C_mean', 'chroma_D_mean', 'chroma_G_mean',
    'tonnetz_1_mean',
    'tempogram_ratio_factor1_mean'
]

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