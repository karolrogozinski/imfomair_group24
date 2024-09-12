import pandas as pd
import numpy as np


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

    def predict(self, X_test: pd.Series) -> list:
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


class BaselineRulebased(Model):
    """ Simple model that returns predictions based on given rules.
    """
    def __init__(self):
        ...

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> None:
        ...

    def predict(self, X_test: pd.Series) -> list:
        y_pred = X_test.apply(lambda x: BaselineRulebased.__predict_sample(x))
        return y_pred

    @staticmethod
    def __predict_sample(x: str) -> str:
        """Predicting sample based on given rules

        Parameters
        ___
        x: str
            sentence to predict

        Return:
        str
            predicted class for given sentence

        [Feel free to change to the rules for better accuracy :)]
        """
        if any(word in x for word in
               ['address', 'phone', 'what is', 'food', 'post']):
            return 'request'
        if any(word in x for word in ['how about', 'else', 'what about']):
            return 'reqalts'
        if any(word in x for word in ['cough', 'unintelligable', 'sil']):
            return 'null'
        if any(word in x for word in ['no']):
            return 'negate'
        if (x.startswith('does') or x.startswith('is')) and\
                x.split()[1] in ['it', 'they', 'that']:
            return 'confirm'
        if any(word in x for word in ['thank']):
            return 'thankyou'
        if any(word in x for word in ['hi', 'hello']):
            return 'hello'
        if any(word in x for word in ['again', 'repeat', 'back']):
            return 'repeat'
        if x == 'more':
            return 'reqmore'
        if any(word in x for word in ['start', 'reset', 'over', 'again']):
            return 'restart'
        if any(word in x for word in ['wrong']):
            return 'deny'
        return 'inform'
