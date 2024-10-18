""" Standard libraries
datime: adding timestamps to evaluations.
"""
from datetime import datetime

""" Third-party libraries
pandas, numpy: data operations
sklearn: evaluation metrics
seaborn, matplotlib: plots, visualizations
"""
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score

import seaborn as sns
import matplotlib.pyplot as plt


class ClassifierEvaluation:
    """Class contains all evaluation metric

    Methods
    accuracy()
        basic evaluation metric given in the assigment
    save_confusion_matrix()
        creates confusion matrix which compares predictions with actual classes and saves to png format
    """
    def __init__(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        self.y_true: pd.Series = y_true
        self.y_pred: pd.Series = y_pred

    def accuracy(self) -> None:
        """Prints the accuracy score of the model."""
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        print(f'Accuracy: {self.accuracy}')

    def precision_recall_f1(self) -> None:
        """Prints the precision, recall, and F1-score for the model."""
        print("Classification Report:")
        self.prec_recall_f1 = classification_report(self.y_true, self.y_pred, zero_division=1)
        self.prec = precision_score(self.y_true, self.y_pred, average="micro", zero_division=1)
        self.recall = recall_score(self.y_true, self.y_pred, average="macro", zero_division=1)
        self.f1_score = f1_score(self.y_true, self.y_pred, average="macro", zero_division=1)
        print(self.prec )
    def save_confusion_matrix(self, model, drop_duplicates) -> None:
        """Creates and saves the confusion matrix as a PNG file."""
        plt.figure(figsize=(10, 8))
        class_names = np.unique(self.y_true)
        matrix = confusion_matrix(self.y_pred, self.y_true, normalize='true')
        sns.heatmap(matrix, annot=True, fmt=".0%", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Actual')
        plt.ylabel('Predictions')

        # Get model name
        if model == "bm":
            model_name = "Baseline Model"
        elif model == "brb":
            model_name = "Base Line Rule Based Model"
        elif model == "fnn":
            model_name = "Feed Forward NN Model"
        else:
            model_name = "Logistic Regression Model"

        # Add title to confusion matrix.
        if drop_duplicates:
            plt.title(f'Confusion matrix for {model_name} without duplicates', fontweight='bold')
        else:
            plt.title(f'Confusion matrix for {model_name} with duplicates', fontweight='bold')
        plt.savefig(f'./reports/eval/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_conf_matrix.png')
        print('Confusion matrix saved to tmp directory.')

    def show_misclassified(self) -> None:
        """Prints the instances where the model misclassified the dialog act."""
        misclassified = pd.DataFrame({'y_true': self.y_true, 'y_pred': self.y_pred})
        misclassified = misclassified[misclassified['y_true'] != misclassified['y_pred']]
        print("Misclassified Instances:")
        print(misclassified)
