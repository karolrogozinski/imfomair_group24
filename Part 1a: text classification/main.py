import argparse
import os
from os.path import abspath, dirname

from src.interface import Interface


if __name__ == '__main__':
    os.chdir(dirname(abspath(__file__)))
    parser = argparse.ArgumentParser()

    # -f FILENAME -m MODEL -dd DROP_DUPLICATES
    parser.add_argument('-f', '--filename', dest='datapath',
                        default='dialog_acts.dat', help='File in data folder in .dat format')
    parser.add_argument('-m', '--model', dest='model_name', default='lr',
                        help="""Model to train and predict:
                                bm: BaselineMajor
                                brb: BaselineRuleBased
                                lr: LogisticRegressiom
                                fnn: FeedForwardNeuralNetwork""")  # TODO add more models after implementation
    parser.add_argument('-dd', '--drop_duplicates', dest='drop_duplicates', default=False, type=bool,
                        help='Drop duplicate entries from data')

    args = parser.parse_args()

    interface = Interface(datapath=args.datapath, model=args.model_name, drop_duplicates=args.drop_duplicates)
    interface.run()
