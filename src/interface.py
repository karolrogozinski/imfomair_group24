import sys
from datetime import datetime

import pandas as pd

from src.utils import prepare_data, get_possible_choices, get_possible_restaurants
from src.models import BaselineMajor, BaselineRuleBased, LogisticRegressorModel, FeedForwardNN
from src.evaluations import ClassifierEvaluation
from src.state_machine import DialogSMLogic

class Interface:
    """
    # TODO add docstring
    """
    def __init__(self, datapath: str, model: str, drop_duplicates: bool, evaluate: bool,
                 task: str, delay: int, tts: bool) -> None:
        self.datapath: str = './data/' + datapath
        self.model_name: str = model
        self.drop_duplicates: bool = drop_duplicates
        self.bow_model = True if model in ('lr', 'fnn') else False
        self.eval: bool = evaluate
        self.task: str = task
        self.delay: int = delay
        self.tts: bool = tts

    def run(self) -> None:
        self.__read_data()
        self.__read_model()
        self.__train_model()
        if self.eval:
            self.__predict()
            self.__evaluate()
        if self.task == '1A':
            Interface.__welcome()
            self.__manual_prediction()
        elif self.task == '1B':
            self.__simple_dialog()
        else:
            pass # TODO part 1C

    @staticmethod
    def __welcome() -> None:
        print('\n###################################')
        print('Welcome in our text classification app!\n')

    def __read_data(self) -> None:
        print('Reading data...')
        try:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test, self.vectorizer = prepare_data(
                path=self.datapath, drop_duplicates=self.drop_duplicates, vectorize=self.bow_model)
        except FileNotFoundError:
            print('File not found!')
            print(self.datapath)
            sys.exit(1)

    def __read_model(self) -> None:
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
        print('Training model... (it may take a while)')
        self.__model.fit(self.__X_train, self.__y_train)

    def __predict(self) -> None:
        print('Predicting...')
        self.__y_pred = self.__model.predict(self.__X_test)

    def __evaluate(self) -> None:
        # TODO separate saving as the Evaluation method?
        print('Evaluating...')
        filename = f'./reports/eval/{datetime.now()}_eval_report.txt'

        stdout_origin = sys.stdout
        sys.stdout = open(filename, "w")

        # Part saved to file
        print('\n-----------------------------------')
        print('MODEL METRICS')
        print(f'Model: {self.model_name}, Drop duplicates: {self.drop_duplicates}')
        evaluation = ClassifierEvaluation(y_pred=self.__y_pred, y_true=self.__y_test)
        evaluation.accuracy()
        evaluation.precision_recall_f1()
        evaluation.save_confusion_matrix()
        print('-----------------------------------')

        sys.stdout.close()
        sys.stdout = stdout_origin
        print(f'Evaluation saved to {filename}')

    @staticmethod
    def __input_sentence():
        print('\nPlease enter your sentence, or type "quit" to exit')
        sentence = input()
        return sentence

    def __manual_prediction(self):
        sentence = Interface.__input_sentence()
        while sentence != 'quit':
            if self.bow_model:
                prediction = self.__model.predict(self.vectorizer.transform(pd.Series(sentence.lower())))
            else:
                prediction = self.__model.predict(pd.Series(sentence.lower()))
            print(f'Prediction: {str(prediction[0])}')
            sentence = Interface.__input_sentence()

    def __simple_dialog(self):
        print('\n###################################')
        possible_choices = get_possible_choices('./data/restaurant_info.csv')
        possible_restaurants = get_possible_restaurants('./data/restaurant_info.csv')

        sm = DialogSMLogic(possible_choices, self.__model, self.vectorizer, possible_restaurants, delay=self.delay,
                           tts=self.tts)
        while True:
            sentence = input('USER: ')
            print('')
            sm.state_transition(sentence)
