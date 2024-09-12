# TODO
# Terminal interface:
# Choosing model
# Predict and display Eval
import sys

from .utils import prepare_data


class Interface:
    def __init__(self, datapath: str, model: str, drop_duplicates: bool, ) -> None:
        self.datapath: str = './data/' + datapath
        self.model_name: str = model
        self.drop_duplicates: bool = drop_duplicates

    def run(self):
        Interface.__welcome()
        self.__read_data()

    @staticmethod
    def __welcome() -> None:
        print('Welcome in our sentence classification application!')

    def __read_data(self):
        print('Reading data...')
        try:
            self.__X_train, self.__y_train, self.__X_test, self.__y_test = prepare_data(
                path=self.datapath, drop_duplicates=self.drop_duplicates)
        except FileNotFoundError:
            print('File not found!')
            print(self.datapath)
            sys.exit(1)
        print('Data read successfully!')
