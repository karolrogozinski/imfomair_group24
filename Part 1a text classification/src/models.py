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
    def __predict_sample(x: str) -> str:
        """Predicting sample based on given rules

        Parameters
        ___
        x: str
            sentence to predict

        Return:
        str
            predicted class for given sentence
        """
        # TODO improve rules for better accuracy
        if any(word in x for word in
               ['address', 'phone', 'what is', 'post']):
            return 'request'
        if x == 'yes':
            return 'affirm'
        if any(word in x for word in ['no']):
            return 'negate'
        if any(word in x for word in ['how about', 'else', 'what about']):
            return 'reqalts'
        if any(word in x for word in ['hi', 'hello', 'halo']):
            return 'hello'
        if any(word in x for word in ['cough', 'unintelligible', 'sil']):
            return 'null'
        if (x.startswith('does') or x.startswith('is')) and\
                x.split()[1] in ['it', 'they', 'that']:
            return 'confirm'
        if any(word in x for word in ['thank']):
            return 'thankyou'
        if any(word in x for word in ['again', 'repeat', 'back']):
            return 'repeat'
        if any(word in x for word in ['kay', 'okay']):
            return 'ack'
        if any(word in x for word in ['bye']):
            return 'bye'
        if x == 'more':
            return 'reqmore'
        if any(word in x for word in ['start', 'reset']):
            return 'restart'
        if any(word in x for word in ['wrong']):
            return 'deny'
        return 'inform'

class LogisticRegressorModel(Model):
    
    def fit(self, X_train: list, y_train: list) -> None:
        # Train the lr model.
        self.lr_model = LogisticRegression(random_state = 42).fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions and return as pandas series.
        self.predictions = self.lr_model.predict(X_test)

        return pd.Series(self.predictions)

class FeedForwardNN(Model):

    def fit(self, X_train: list, y_train: list) -> None:
        # Train the fnn model. Do we need to make hyperparameter tuning?
            # The project description does not ask us to divide development data for such hyper parameter tuning.
        self.fnn_model = MLPClassifier(random_state=42, max_iter=300, solver="adam").fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions and return as pandas series.
        self.predictions = self.fnn_model.predict(X_test)

        return pd.Series(self.predictions)


