from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt


class Evaluation:
    """Class contains all evaluation metric

    Methods
    accuracy()
        basic evaluation metric given in the assigment
    save_confusion_matrix()
        creates confusion matrix which compares predictions with actual classes and saves to png format
    """
    # TODO Implement more evaluations

    def __init__(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        self.y_true: pd.Series = y_true
        self.y_pred: pd.Series = y_pred

    def accuracy(self) -> None:
        """Prints the accuracy score of the model."""
        print(f'Accuracy: {accuracy_score(self.y_true, self.y_pred)}')

    def precision_recall_f1(self) -> None:
        """Prints the precision, recall, and F1-score for the model."""
        print("Classification Report:")
        print(classification_report(self.y_true, self.y_pred, zero_division=1))

    def save_confusion_matrix(self) -> None:
        """Creates and saves the confusion matrix as a PNG file."""
        plt.figure(figsize=(10, 8))
        class_names = np.unique(self.y_true)
        matrix = confusion_matrix(self.y_pred, self.y_true, normalize='true')
        sns.heatmap(matrix, annot=True, fmt=".0%", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Actual')
        plt.ylabel('Predictions')
        # plt.savefig(f'./tmp/conf_matrix_{datetime.now()}.png')
        print('Confusion matrix saved to tmp directory.')

    def show_misclassified(self) -> None:
        """Prints the instances where the model misclassified the dialog act."""
        misclassified = pd.DataFrame({'y_true': self.y_true, 'y_pred': self.y_pred})
        misclassified = misclassified[misclassified['y_true'] != misclassified['y_pred']]
        print("Misclassified Instances:")
        print(misclassified)