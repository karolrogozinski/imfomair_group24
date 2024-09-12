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
        print(f'Accuracy: {accuracy_score(self.y_true, self.y_pred)}')

    def save_confusion_matrix(self) -> None:
        plt.figure(figsize=(10, 8))

        class_names = np.unique(self.y_true)
        matrix = confusion_matrix(self.y_pred, self.y_true, normalize='true')
        sns.heatmap(matrix, annot=True, fmt=".0%", cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

        plt.xlabel('Actual')
        plt.ylabel('Predictions')

        plt.savefig(f'./tmp/conf_matrix_{datetime.now()}.png')
        print('Confusion matrix saved to tmp directory.')
