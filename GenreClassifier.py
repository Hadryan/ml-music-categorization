import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize

from constants import RANDOM_SEED


class GenreClassifier:
    def __init__(self, X_train_bal, y_train_bal, X_test, y_test, label_encoder, top_features):
        self.X_train_bal = X_train_bal
        self.y_train_bal = y_train_bal
        self.X_test = X_test
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.top_features = top_features
        self.best_model = None

    def tune_model_with_grid_search(self, param_grid, cv, scoring, n_jobs):
        """
        Fits a GridSearchCV with a RandomForestClassifier and stores the best model.
        """
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced'),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        grid_search.fit(self.X_train_bal, self.y_train_bal)
        self.best_model = grid_search.best_estimator_

    def evaluate_model(self):
        """
        Prints a classification report comparing predicted vs. actual on X_test, y_test.
        """
        y_pred = self.best_model.predict(self.X_test)
        print("Classification Report:")
        print(classification_report(
            self.y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))

    def plot_feature_importance(self):
        """
        Plots a horizontal bar chart of feature importances according to the final model.
        """
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [self.top_features[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title("Top Feature Importances in Final Model")
        plt.show()

    def plot_pairwise_relationships(self, top_n=2):

        df_train_bal = pd.DataFrame(self.X_train_bal, columns=self.top_features)
        df_train_bal['genre'] = self.label_encoder.inverse_transform(self.y_train_bal)

        # Pairplot
        sns.pairplot(
            df_train_bal,
            vars=self.top_features[:top_n],
            hue='genre',
            height=4,
            palette='tab10'
        )
        plt.suptitle("Pairwise Feature Relationships", y=1.02)
        plt.show()

    def plot_precision_recall_curve(self):
        """
        Plots Precision-Recall curves for each class in a multi-class setting.
        """
        # Binarize the test labels for each class
        n_classes = len(self.label_encoder.classes_)
        y_test_bin = label_binarize(self.y_test, classes=range(n_classes))

        # Probability predictions
        y_pred_proba = self.best_model.predict_proba(self.X_test)

        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(self.label_encoder.classes_):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, label=class_name)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="best")
        plt.show()
