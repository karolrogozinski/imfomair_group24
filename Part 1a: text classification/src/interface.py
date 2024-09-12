import sys

import pandas as pd

from .utils import prepare_data
from .models import BaselineMajor, BaselineRuleBased
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

    def __read_model(self) -> None:
        # TODO add more models after implementation
        print('Reading model...')
        if self.model_name == 'bm':
            self.__model = BaselineMajor()
        elif self.model_name == 'brb':
            self.__model = BaselineRuleBased()
        else:
            print('Model not found!')
            sys.exit(2)

    def __train_model(self) -> None:
        print('Training model...')
        self.__model.fit(self.__X_train, self.__y_train)

    def __predict(self) -> None:
        print('Predicting...')
        self.__y_pred = self.__model.predict(self.__X_test)

    def __evaluate(self) -> None:
        print('Evaluating...')
        print('\n-----------------------------------')
        print('MODEL METRICS')
        evaluation = Evaluation(y_pred=self.__y_pred, y_true=self.__y_test)
        evaluation.accuracy()
        # TODO add more evaluations after implementation
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
            prediction = self.__model.predict(pd.Series(sentence.lower()))
            print(f'Prediction: {str(prediction[0])}')
            sentence = Interface.__input_sentence()
