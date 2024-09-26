import argparse
import os
from os.path import abspath, dirname

from src.interface import Interface


if __name__ == '__main__':

    os.chdir(dirname(abspath(__file__)))
    parser = argparse.ArgumentParser()

    # -f FILENAME -m MODEL -dd DROP_DUPLICATES -e EVALUATE
    parser.add_argument('-f', '--filename', dest='datapath',
                        default='dialog_acts.dat', help='File in data folder in .dat format')
    parser.add_argument('-t', '--task', dest='task',
                        default='1B', help='Project subpart to run')
    parser.add_argument('-m', '--model', dest='model_name', default='fnn',
                        help="""Model to train and predict:
                                bm: BaselineMajor
                                brb: BaselineRuleBased
                                lr: LogisticRegressio
                                fnn: FeedForwardNeuralNetwork""")
    parser.add_argument('-rd', '--response_delay', dest='response_delay', default=0, type=int,
                        help='Add delay before system responses in (s)')
    parser.add_argument('-dd', '--drop_duplicates', dest='drop_duplicates', action='store_true',
                        help='Drop duplicate entries from data')
    parser.add_argument('-e', '--evaluate', dest='evaluate',action='store_true',
                        help='Make evaluation and save it to file')

    args = parser.parse_args()

    args.drop_duplicates = True if args.drop_duplicates == 'True' else False

    interface = Interface(datapath=args.datapath, model=args.model_name, drop_duplicates=args.drop_duplicates,
                          evaluate=args.evaluate, task=args.task, delay=args.response_delay)
    interface.run()
