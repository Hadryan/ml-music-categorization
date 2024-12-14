import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

RANDOM_SEED = 42

# Random Forest Parameters
RANDOM_FOREST_N_ESTIMATORS = 100  # Number of trees in the Forest
RANDOM_FOREST_MAX_DEPTH = 15   # Maximum depth of each tree
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 2  # Minimum samples to split a node

TRAIN_TEST_SPLIT_RATIO = 0.2  # Test size
FEATURE_SELECTION_TOP_N = 50  # Number of top features to select based on importance
CLASS_MIN_SAMPLES = 40        # Minimum samples a class must have to be included in the model

# Grid Search Parameters
GRID_SEARCH_CV = 3            # Number of cross-validation folds
GRID_SEARCH_N_JOBS = -1       # Number of jobs to run in parallel (-1 uses all cores)

print("Training model with the following parameters:")
print("-" * 50)
print(f"{'RANDOM_FOREST_N_ESTIMATORS':<30} : {RANDOM_FOREST_N_ESTIMATORS}")
print(f"{'RANDOM_FOREST_MAX_DEPTH':<30} : {RANDOM_FOREST_MAX_DEPTH}")
print(f"{'RANDOM_FOREST_MIN_SAMPLES_SPLIT':<30} : {RANDOM_FOREST_MIN_SAMPLES_SPLIT}")
print(f"{'TRAIN_TEST_SPLIT_RATIO':<30} : {TRAIN_TEST_SPLIT_RATIO}")
print(f"{'FEATURE_SELECTION_TOP_N':<30} : {FEATURE_SELECTION_TOP_N}")
print(f"{'CLASS_MIN_SAMPLES':<30} : {CLASS_MIN_SAMPLES}")
print(f"{'GRID_SEARCH_CV':<30} : {GRID_SEARCH_CV}")
print(f"{'GRID_SEARCH_N_JOBS':<30} : {GRID_SEARCH_N_JOBS}")
print("-" * 50)

# Load data
df = pd.read_csv('data/track_features.csv')

# Drop unimportant features 
X = df.drop(columns=['artist', 'title', 'genre'])

# Identify Target
y = df['genre']

# Much of the genres have very few tracks, 
# Lets just exclude these completely and filter classes with fewer than CLASS MIN samples
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= CLASS_MIN_SAMPLES].index
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=y_encoded
)


# The classes are very imbalenced, lets use SMOTE
# This will create new samples for underrepresented classes
# It does this by generating a sample at a point between two exsitsing ones.
sm = SMOTE(random_state=RANDOM_SEED)
X_train_bal, y_train_bal = sm.fit_resample(X_train.values, y_train)

# Train a random forrest classifier, we will use this to help slect the most important features
temp_rf = RandomForestClassifier(n_estimators=RANDOM_FOREST_N_ESTIMATORS, random_state=RANDOM_SEED)
temp_rf.fit(X_train_bal, y_train_bal)

# Get an array of feature importances based on the random forest
importances = temp_rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
top_features = X.columns[sorted_indices[:FEATURE_SELECTION_TOP_N]]

# Reduce data to top features
X_train_bal_top = X_train_bal[:, sorted_indices[:FEATURE_SELECTION_TOP_N]]
X_test_top = X_test.values[:, sorted_indices[:FEATURE_SELECTION_TOP_N]]

# Perform GridSearch for Random Forest hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced'),
    param_grid=param_grid,
    cv=GRID_SEARCH_CV,
    scoring='accuracy',
    n_jobs=GRID_SEARCH_N_JOBS
)
grid_search.fit(X_train_bal_top, y_train_bal)

# Evaluate the final model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_top)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))


# Plot feature importance
final_importances = best_model.feature_importances_
indices = np.argsort(final_importances)
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), final_importances[indices], align='center')
plt.yticks(range(len(indices)), [top_features[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top Feature Importances in Final Model')
plt.show()


top_2_features = top_features[:2]
df_train_bal = pd.DataFrame(X_train_bal[:, sorted_indices[:len(top_features)]], columns=top_features)
df_train_bal['genre'] = label_encoder.inverse_transform(y_train_bal)

sns.pairplot(df_train_bal, vars=top_2_features, hue='genre', height=4, palette='tab10')
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()


y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
y_pred_proba = best_model.predict_proba(X_test_top)

# Plot precision-recall curves for each class
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(label_encoder.classes_):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(recall, precision, label=class_name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()