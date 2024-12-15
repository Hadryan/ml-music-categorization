# ml-music-categorization

A machine learning project for classifying and clustering music tracks based on extracted audio features. Designed to assist music professionals in organizing and analyzing their libraries.

## Requirements

- **Python Version:** 3.9 (recommended to use a Conda virtual environment)
- **Dependencies:**
  - `numpy`
  - `pandas`
  - `librosa`
  - `mutagen`
  - `scikit-learn`
  - `imbalanced-learn`
  - `matplotlib`
  - `seaborn`

Install the required packages using:

```bash
pip install numpy pandas librosa mutagen scikit-learn imbalanced-learn matplotlib seaborn
```

If using a Conda environment:

```bash
conda create -n ml-music python=3.9
conda activate ml-music
pip install numpy pandas librosa mutagen scikit-learn imbalanced-learn matplotlib seaborn
```

## Files

### 1. `TrackAnalyzer.py`

A class for extracting audio features from MP3 files using **librosa** and **mutagen**. Features extracted include tempo, spectral features, chroma, MFCCs, tonal centroid (Tonnetz), and more.

#### Usage Example:

```python
from TrackAnalyzer import TrackAnalyzer

analyzer = TrackAnalyzer("path/to/audio/file.mp3")
analyzer.extract_features()
features = analyzer.get_features()
print(features)
```

---

### 2. `extract_data.py`

A script for batch-processing MP3 files in a specified directory. It extracts features using the `TrackAnalyzer` class and saves the data to `data/track_features.csv` in CSV format.

#### Usage Example:

```bash
python extract_data.py
```

---

### 3. `supervised_model.py`

A script for building a supervised machine learning model to classify music tracks based on their genres. It uses a **Random Forest Classifier** and includes:

- **SMOTE** for handling class imbalance.
- Feature selection based on feature importance.
- **Grid Search** for hyperparameter tuning.
- Visualization of feature importance, precision-recall curves, and relationships between top features.

#### How It Works:

1. Loads the feature data from `data/track_features.csv`.
2. Prepares the data by filtering out classes with fewer than a configurable minimum number of samples.
3. Uses SMOTE to balance the training dataset.
4. Trains and evaluates a Random Forest model, performing hyperparameter optimization using Grid Search.
5. Visualizes top features and their relationships.

#### Usage Example:

```bash
python supervised_model.py
```

---

### 4. `unsupervised_model.py`

A script for clustering music tracks into groups based on extracted features. It includes:

- **Dimensionality Reduction** using either PCA or Recursive Feature Elimination (RFE).
- Cluster evaluation using **Elbow Method** (WCSS) and **Silhouette Scores**.
- **KMeans Clustering** for grouping tracks.
- Visualization of clusters and sorted playlist creation.

#### Features:

- Sorts clusters based on centroid similarity.
- Orders tracks within each cluster by nearest neighbor distance.
- Outputs a CSV file `clustered_playlist.csv` containing the sorted playlist.

#### How It Works:

1. Loads feature data from `data/track_features.csv`.
2. Pre-processes the data by normalizing features and removing irrelevant/correlated features.
3. Applies PCA or RFE for dimensionality reduction.
4. Evaluates cluster quality and performs KMeans clustering.
5. Sorts clusters and tracks to generate a structured playlist.

#### Usage Example:

```bash
python unsupervised_model.py
```

---

## Outputs

### 1. **Feature Extraction Output**

The `extract_data.py` script saves extracted features to `data/track_features.csv` in the following format:

```
Artist,Title,Genre,Tempo,RMS_mean,RMS_std,...,MFCC_1_mean,MFCC_1_std,...,Chroma_C_mean,Chroma_C_std,...
Artist1,Title1,Genre1,120.0,0.123,0.012,...,10.1,1.1,...,0.5,0.05,...
Artist2,Title2,Genre2,128.0,0.150,0.014,...,11.2,1.2,...,0.6,0.06,...
```

### 2. **Supervised Model Output**

The `supervised_model.py` script:

- Prints a classification report for genre prediction.
- Displays visualizations of feature importance, precision-recall curves, and top feature relationships.

### 3. **Unsupervised Model Output**

The `unsupervised_model.py` script:

- Outputs `clustered_playlist.csv` with the following columns:
  ```
  Artist,Title,Genre,Cluster,Track_Index,NN_Distance
  ```
- Generates visualizations of clusters, explained variance (PCA), and feature correlations.

---

## Notes

- Ensure the `data/` directory exists in the project root for saving and processing CSV files.
- Supervised model training requires a sufficient number of labeled samples for each genre.
- For clustering, use the `K` parameter in `unsupervised_model.py` to specify the number of clusters.
