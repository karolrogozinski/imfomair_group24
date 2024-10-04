import argparse
import os
from os.path import abspath, dirname

from src.interface import Interface


if __name__ == '__main__':

    os.chdir(dirname(abspath(__file__)))
    parser = argparse.ArgumentParser()

    # -f FILENAME -m MODEL -rd RESPONSE_DELAY -dd DROP_DUPLICATES -e EVALUATE -tts TEXT_TO_SPEECH -asr AUTOMATIC_SPEECH_RECOGNITION
    parser.add_argument('-f', '--filename', dest='datapath',
                        default='dialog_acts.dat', help='File in data folder in .dat format')
    parser.add_argument('-t', '--task', dest='task',
                        default='1B', help='Project subpart to run')
    parser.add_argument('-m', '--model', dest='model_name', default='brb',
                        help="""Model to train and predict:
                                bm: BaselineMajor
                                brb: BaselineRuleBased
                                lr: LogisticRegression
                                fnn: FeedForwardNeuralNetwork""")
    parser.add_argument('-rd', '--response_delay', dest='response_delay', default=0, type=int,
                        help='Add delay before system responses in (s)')
    parser.add_argument('-dd', '--drop_duplicates', dest='drop_duplicates', action='store_true',
                        help='Drop duplicate entries from data')
    parser.add_argument('-e', '--evaluate', dest='evaluate',action='store_true',
                        help='Make evaluation and save it to file')
    parser.add_argument('-tts', '--text_to_speech', dest='tts',action='store_true',
                        help='Use text-to-speech for system utterances')
    parser.add_argument('-asr', '--automatic_speech_recognition', dest='asr',action='store_true',
                        help='Use automatic speech recognition (ASR) for user utterances')
    parser.add_argument('-ht', '--hyper_param_tuning', dest='hyper_param_tuning', action='store_true')

    args = parser.parse_args()

    args.drop_duplicates = True if args.drop_duplicates == 'True' else False
    args.hyper_param_tuning = True if args.hyper_param_tuning == 'True' else False

    #args.hyper_param_tuning = True
    #args.drop_duplicates = True

    interface = Interface(datapath=args.datapath, model=args.model_name, drop_duplicates=args.drop_duplicates,
                          evaluate=args.evaluate, task=args.task, hyper_param_tuning=args.hyper_param_tuning, 
                          delay=args.response_delay, tts=args.tts, asr=args.asr)
    interface.run()
