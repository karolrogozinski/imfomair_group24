""" Third-party libraries
pandas, numpy: data operations
sklearn: ml models implementations
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class Model:
    """Abstract class for Model.
    Contains methods that should be inherited and developed in every model.

    Methods
    ---
    fit(X_train: np.array, y_train: np.array)
        basic method for fitting the model to the training data
    predict(X_test: np.array) -> list
        predicting classes of provided list of sentences
    """
    def fit(self, X_train: list, y_train: list) -> None:
        ...

    def predict(self, X_test: list) -> list:
        ...


class BaselineMajor(Model):
    """Very simple model that returns the most common class from the
    training data.

    Attributes
    ---
    prediction: str
        variable holding the most common class
    """
    def __init__(self):
        self.prediction: str = None

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> None:
        self.prediction = y_train.mode()

    def predict(self, X_test: pd.Series) -> np.array:
        y_pred = np.array([self.prediction]).repeat(X_test.shape[0])
        return y_pred


class BaselineRuleBased(Model):
    """ Simple model that returns predictions based on given rules.
    """
    def __init__(self):
        ...

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> None:
        ...

    def predict(self, X_test: pd.Series) -> list:
        y_pred = X_test.apply(lambda x: BaselineRuleBased.__predict_sample(x))
        return y_pred

    @staticmethod
    def __predict_sample(sentence: str) -> str:
        """Predicting sample based on given rules

        Parameters
        ___
        x: str
            sentence to predict

        Return:
        str
            predicted class for given sentence
        """
        request_keywords = ['address', 'phone', 'what is', 'post']
        negate_keywords = ['no']
        reqalts_keywords = ['how about', 'else', 'what about']
        hello_keywords = ['hi', 'hello', 'halo']
        null_keywords = ['cough', 'unintelligible', 'sil']
        thankyou_keywords = ['thank']
        repeat_keywords = ['again', 'repeat', 'back']
        ack_keywords = ['kay', 'okay']
        bye_keywords = ['bye']
        restart_keywords = ['start', 'reset']
        deny_keywords = ['wrong']

        if any(word in sentence for word in request_keywords):
            return 'request'
        elif sentence == 'yes':
            return 'affirm'
        elif any(word in sentence for word in negate_keywords):
            return 'negate'
        elif any(word in sentence for word in reqalts_keywords):
            return 'reqalts'
        elif any(word in sentence for word in hello_keywords):
            return 'hello'
        elif any(word in sentence for word in null_keywords):
            return 'null'
        elif (sentence.startswith('does') or sentence.startswith('is')) and \
                sentence.split()[1] in ['it', 'they', 'that']:
            return 'confirm'
        elif any(word in sentence for word in thankyou_keywords):
            return 'thankyou'
        elif any(word in sentence for word in repeat_keywords):
            return 'repeat'
        elif any(word in sentence for word in ack_keywords):
            return 'ack'
        elif any(word in sentence for word in bye_keywords):
            return 'bye'
        elif sentence == 'more':
            return 'reqmore'
        elif any(word in sentence for word in restart_keywords):
            return 'restart'
        elif any(word in sentence for word in deny_keywords):
            return 'deny'
        else:
            return 'inform'

class LogisticRegressorModel(Model):
    def fit(self, X_train: list, y_train: list, solver="saga", penalty="l2", C=10, max_iter=200) -> None:
        # Train the lr model.
        self.lr_model = LogisticRegression(random_state = 42, solver=solver, penalty=penalty, C=C,
                                           max_iter=max_iter).fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions and return as pandas series.
        predictions = self.lr_model.predict(X_test)

        return pd.Series(predictions)

class FeedForwardNN(Model):
    def fit(self, X_train: list, y_train: list, activation="relu", solver="adam", batch_size=200, alpha=0.00001,
            max_iter=200, hidden_layer_sizes=250) -> None:
        # Train the fnn model. Do we need to make hyperparameter tuning?
            # The project description does not ask us to divide development data for such hyper parameter tuning.
        self.fnn_model = MLPClassifier(random_state=42, max_iter=max_iter, solver=solver, activation=activation,
                                       batch_size=batch_size, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes).fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions and return as pandas series.
        self.predictions = self.fnn_model.predict(X_test)

        return pd.Series(self.predictions)
