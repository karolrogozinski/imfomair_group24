import sys

import pandas as pd

from .utils import prepare_data, prepare_data_bow
from .models import BaselineMajor, BaselineRuleBased, LogisticRegressorModel, FeedForwardNN
from .evaluation import Evaluation


class Interface:
    """
    # TODO add docstring
    """
    def __init__(self, datapath: str, model: str, drop_duplicates: bool, ) -> None:
        self.datapath: str = './data/' + datapath
        self.model_name: str = model
        self.drop_duplicates: bool = drop_duplicates

    def run(self) -> None:
        Interface.__welcome()
        self.__read_data()
        self.__read_data_bow()
        self.__read_model()
        self.__train_model()
        self.__predict()
        self.__evaluate()
        self.__manual_prediction()

    @staticmethod
    def __welcome() -> None:
        print('\n###################################')
        print('Welcome in our text classification app!\n')

    def __read_data(self) -> None:
        print('Reading data...')
        try:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = prepare_data(
                path=self.datapath, drop_duplicates=self.drop_duplicates)
        except FileNotFoundError:
            print('File not found!')
            print(self.datapath)
            sys.exit(1)

    def __read_data_bow(self) -> None:
        # Implemented this in a different function because numeric BoW data is completely different from the
            # string data the baseline models use.
            # We will need the vectorizer to vectorize single input sentences coming from the console.
        print("Reading data in BoW format...")
        try:
            self.__X_train_bow, self.__X_test_bow, self.__y_train_bow, self.__y_test_bow, self.vectorizer = prepare_data_bow(
                path=self.datapath, drop_duplicates=self.drop_duplicates)
        except FileNotFoundError:
            print('File not found!')
            print(self.datapath)
            sys.exit(1)

    def __read_model(self) -> None:
        # TODO add more models after implementation
        print('Reading model...')
        if self.model_name == 'bm':
            self.__model = BaselineMajor()
        elif self.model_name == 'brb':
            self.__model = BaselineRuleBased()
        elif self.model_name == 'lr':
            self.__model = LogisticRegressorModel()
        elif self.model_name == 'fnn':
            self.__model = FeedForwardNN()
        else:
            print('Model not found!')
            sys.exit(2)

    def __train_model(self) -> None:
        print('Training model...')
        # if the model is logistic regression of feed forward nn, then we use BoW data
        if self.model_name in ('lr', 'fnn'):
            self.__model.fit(self.__X_train_bow, self.__y_train_bow)
        else:
            self.__model.fit(self.__X_train, self.__y_train)

    def __predict(self) -> None:
        print('Predicting...')
        # if the model is logistic regression of feed forward nn, then we use BoW data
        if self.model_name in ('lr', 'fnn'):
            self.__y_pred = self.__model.predict(self.__X_test_bow)
        else:
            self.__y_pred = self.__model.predict(self.__X_test)

    def __evaluate(self) -> None:
        print('Evaluating...')
        print('\n-----------------------------------')
        print('MODEL METRICS')
        evaluation = Evaluation(y_pred=self.__y_pred, y_true=self.__y_test)
        evaluation.accuracy()
        # TODO add more evaluations after implementation
        evaluation.precision_recall_f1()
        evaluation.save_confusion_matrix()
        print('\n-----------------------------------')

    @staticmethod
    def __input_sentence():
        print('Please enter your sentence, or type "quit" to exit')
        sentence = input()
        return sentence

    def __manual_prediction(self):
        sentence = Interface.__input_sentence()
        while sentence != 'quit':
            # if we are to use BoW data with lr or fnn, then we need to vectorize the console input too.
            if self.model_name in ('lr', 'fnn'):
                prediction = self.__model.predict(self.vectorizer.transform(pd.Series(sentence.lower())))
            else:    
                prediction = self.__model.predict(pd.Series(sentence.lower()))
            print(f'Prediction: {str(prediction[0])}')
            sentence = Interface.__input_sentence()
