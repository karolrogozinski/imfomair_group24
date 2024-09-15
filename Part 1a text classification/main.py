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
    parser.add_argument('-m', '--model', dest='model_name', default='fnn',
                        help="""Model to train and predict:
                                bm: BaselineMajor
                                brb: BaselineRuleBased
                                lr: LogisticRegressiom
                                fnn: FeedForwardNeuralNetwork""")
    parser.add_argument('-dd', '--drop_duplicates', dest='drop_duplicates', default=True, type=bool,
                        help='Drop duplicate entries from data')

    args = parser.parse_args()

    # Latest one, if you want to run this comment out the others and run this one
    # interface = Interface(datapath=args.datapath, model=args.model_name, drop_duplicates=args.drop_duplicates)
    # interface.run()

    # # Run BaselineMajor with duplication
    # interface = Interface(datapath='dialog_acts.dat', model='bm', drop_duplicates=False)
    # interface.run()

    # Run BaselineMajor without duplication(unique)
    # interface = Interface(datapath='dialog_acts.dat', model='bm', drop_duplicates=True)
    # interface.run()

    # Run BaselineRuleBased with duplication
    interface = Interface(datapath='dialog_acts.dat', model='brb', drop_duplicates=False)
    interface.run()

    # Run BaselineRuleBased without duplication
    # interface = Interface(datapath='dialog_acts.dat', model='brb', drop_duplicates=True)
    # interface.run()

    # Run Logistic Regression with duplication
    # interface = Interface(datapath='dialog_acts.dat', model='lr', drop_duplicates=False)
    # interface.run()

    # Run Logistic Regression without duplication
    # interface = Interface(datapath='dialog_acts.dat', model='lr', drop_duplicates=True)
    # interface.run()

    # Run FeedForward Neural Network with duplication
    # interface = Interface(datapath='dialog_acts.dat', model='fnn', drop_duplicates=False)
    # interface.run()

    # Run FeedForward Neural Network without duplication
    # interface = Interface(datapath='dialog_acts.dat', model='fnn', drop_duplicates=True)
    # interface.run()

