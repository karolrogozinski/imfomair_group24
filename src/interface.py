""" Standard libraries
sys: basic system operations, like closing app
datetime: adding timestamps
"""
import sys
from datetime import datetime

""" Third-party libraries
pandas: data operations
"""
import pandas as pd

""" Local files
utils: basic utility functions
models: classes containing models for speech acts classification
evaluations: models evaluations
state_machine: main dialog state machine
"""
from src.utils import prepare_data, get_possible_choices, get_possible_restaurants, automatic_speech_recognition
from src.models import BaselineMajor, BaselineRuleBased, LogisticRegressorModel, FeedForwardNN
from src.evaluations import ClassifierEvaluation
from src.state_machine import DialogSMLogic


class Interface:
    """
    # TODO add docstring
    """
    def __init__(self, datapath: str, model: str, drop_duplicates: bool, evaluate: bool,
                 task: str, delay: int, tts: bool, asr: bool, hyper_param_tuning: bool) -> None:
        self.datapath: str = './data/' + datapath
        self.model_name: str = model
        self.drop_duplicates: bool = drop_duplicates
        self.bow_model = True if model in ('lr', 'fnn') else False
        self.eval: bool = evaluate
        self.task: str = task
        self.hyper_param_tuning: bool = hyper_param_tuning
        self.delay: int = delay
        self.tts: bool = tts
        self.asr: bool = asr

    def run(self) -> None:
        self.__read_data()
        self.__read_model()

        # if we are going to perform hyper param tuning or the regular execution pattern
        if self.hyper_param_tuning:
            self.__hyper_param_tuning()
        else:
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
                # Currently 1C part gets the same dialog act as for 1B (it was further develop to new match requirements)
                pass

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

    def __hyper_param_tuning(self) -> None:
        print('Hyper parameters are on trial. (this will take quiet a while.)')

        if self.model_name == "fnn":

            # Parameters: activation="relu", solver="adam", batch_size=200, alpha=0.001, max_iter=200, hidden_layer_size
            # We are going to iterate these values and collect relevant scores: accuracy, recall, precision, f1
            activation_list = ["logistic", "tanh", "relu"]
            solver_list = ["lbfgs", "sgd", "adam"]
            batch_size_list = [200, 400, 1000]
            alpha_list = [10**(-5), 10**(-4), 10**(-3)]
            max_iter_list = [200, 500, 1000]
            hidden_layer_size_list = [10, 100, 250]

            # Dataframe that holds the results
            results_df = pd.DataFrame(columns=['Activation', 'Solver', 'Batch Size', 'Alpha', 'Max Iter', 'Hidden Layer Size', 'Accuracy', 'Precision', 'Recall', 'F1'])

            train_count = 0
            total_training_needed = len(activation_list) * len(solver_list) * len(batch_size_list) * len(alpha_list) * len(max_iter_list)

            for activation in activation_list:
                for solver in solver_list:
                    for batch_size in batch_size_list:
                        for alpha in alpha_list:
                            for max_iter in max_iter_list:
                                for hidden_layer_size in hidden_layer_size_list:
                                    # Work the model
                                    self.__model.fit(X_train=self.__X_train, y_train=self.__y_train, activation=activation, solver=solver, batch_size=batch_size, alpha=alpha, max_iter=max_iter, hidden_layer_sizes=hidden_layer_size)
                                    self.__y_pred = self.__model.predict(self.__X_test)

                                    # Evaluate the model
                                    evaluation = ClassifierEvaluation(y_pred=self.__y_pred, y_true=self.__y_test)
                                    evaluation.accuracy()
                                    evaluation.precision_recall_f1()

                                    train_count += 1
                                    print(f"Training_count: {train_count} / {total_training_needed}")

                                    # Collect the scores
                                    # Create a new row as a DataFrame
                                    new_row = pd.DataFrame({
                                        'Activation': [activation],
                                        'Solver': [solver],
                                        'Batch Size': [batch_size],
                                        'Alpha': [alpha],
                                        'Max Iter': [max_iter],
                                        'Hidden Layer Size': [hidden_layer_size],
                                        'Accuracy': [evaluation.accuracy],
                                        'Precision': [evaluation.prec],
                                        'Recall': [evaluation.recall],
                                        'F1': [evaluation.f1_score]
                                    })

                                    # Concatenate the new row with the results DataFrame
                                    results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            # Put the results into an Excel file
            results_df.to_excel("./reports/eval/model_training_results_fnn.xlsx", index=False)
        elif self.model_name == "lr":
            # Hyperparameters for Logistic Regression
            solver_list = ["lbfgs", "newton-cg", "sag", "saga"]
            penalty_list = ["l2"]
            C_list = [0.001, 0.01, 0.1, 1, 10, 100]
            max_iter_list = [100, 200, 500]

            # df to hold results
            results_df = pd.DataFrame(columns=['Solver', 'Penalty', 'C', 'Max Iter', 'Accuracy', 'Precision', 'Recall', 'F1'])

            train_count = 0
            total_training_needed = len(solver_list) * len(penalty_list) * len(C_list) * len(max_iter_list)

            for solver in solver_list:
                for penalty in penalty_list:
                    for C in C_list:
                        for max_iter in max_iter_list:
                            # Train the Logistic Regression model with given parameters
                            self.__model.fit(self.__X_train, self.__y_train, solver=solver, penalty=penalty, C=C, max_iter=max_iter)
                            self.__y_pred = self.__model.predict(self.__X_test)

                            # Evaluate the model
                            evaluation = ClassifierEvaluation(y_pred=self.__y_pred, y_true=self.__y_test)
                            evaluation.accuracy()
                            evaluation.precision_recall_f1()

                            train_count += 1
                            print(f"Training_count: {train_count} / {total_training_needed}")

                            # Collect the scores
                            new_row = pd.DataFrame({
                                'Solver': [solver],
                                'Penalty': [penalty],
                                'C': [C],
                                'Max Iter': [max_iter],
                                'Accuracy': [evaluation.accuracy],
                                'Precision': [evaluation.prec],
                                'Recall': [evaluation.recall],
                                'F1': [evaluation.f1_score]
                            })

                            # Concatenate the new row with the results DataFrame
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            # Save the results to an Excel file
            results_df.to_excel("./reports/eval/model_training_results_lr.xlsx", index=False)




    def __train_model(self) -> None:
        print('Training model... (it may take a while)')
        self.__model.fit(self.__X_train, self.__y_train)

    def __predict(self) -> None:
        print('Predicting...')
        self.__y_pred = self.__model.predict(self.__X_test)

    def __evaluate(self) -> None:
        # TODO separate saving as the Evaluation method?
        print('Evaluating...')
        filename = f'./reports/eval/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_eval_report.txt'

        stdout_origin = sys.stdout
        sys.stdout = open(filename, "w")

        # Part saved to file
        print('\n-----------------------------------')
        print('MODEL METRICS')
        print(f'Model: {self.model_name}, Drop duplicates: {self.drop_duplicates}')
        evaluation = ClassifierEvaluation(y_pred=self.__y_pred, y_true=self.__y_test)
        evaluation.accuracy()
        evaluation.precision_recall_f1()
        evaluation.save_confusion_matrix(self.model_name, self.drop_duplicates)
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
        possible_choices = get_possible_choices('./data/restaurant_info_copy_.csv')
        possible_restaurants = get_possible_restaurants('./data/restaurant_info_copy_.csv')

        sm = DialogSMLogic(possible_choices, self.__model, self.vectorizer, possible_restaurants, delay=self.delay,
                           tts=self.tts)
        while True:
            if self.asr:
                sentence = automatic_speech_recognition()
                print('USER:', sentence)
            else:
                sentence = input('USER: ')

            print('')
            if sentence:
                sm.state_transition(sentence)
