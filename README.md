# Restaurant reccomendations dialog system
Project consists of a design, implementation, evaluatation and reports about a restaurant recommendations dialog system using various AI methods, such as domain modeling, text classification using machine learning and user experience testing. 

## Table of Contents

1. [Purpose of the project](#purpose-of-the-project)
2. [Requirements](#requirements)
3. [Run](#run)
4. [Data](#data)
5. [Code Structure](#code-structure)
6. [Experiments](#experiments)
7. [References](#references)

## Purpose of the project
Project made for Utrecht University course [[1]](#references).
It is divided into two parts:

1. The first part of the project concerns the implementation of the dialog system: modeling the domain in a dialog model, implementing and evaluating a machine learning classifier for natural language, and developing a text-based dialog system application based on the dialog model.
2. The second part of the project is about evaluating system: designing, carrying out and reporting on user experiments, as well as thinking about your system in the wider context of AI.

### Learning goals for Part 1:
 - Understanding and modeling a specific knowledge domain
 - Implementing and empirically evaluating a machine learning-based NLP algorithm
 - Implement a working AI system using Python
 - Writing a technical report about an AI system and its performance

### Learning goals for Part 2:
 - Designing an experiment with human participants as a way to test a hypothesis that follows from a research question
 - Conducting an experiment with human participants as a way to test a hypothesis (and experiencing the difficulty of collecting good data, and why you need to think about this hard)
 - Analyzing empirical data using statistical techniques
 - Writing a scientific report about your system's empirical evaluation and its place in AI

## Requirements
The project was developed and tested with ```python 3.9.12``` and libraries from [requirements.txt](requirements.txt):

```txt
- pandas=1.4.4
- scikit-learn=1.0.2
- numpy=1.23.5
- seaborn=0.11.2
- matplotlib=3.7.0
- textdistance=4.2.1
```

For the ASR functionality internet connection is also required.

## Run
The easier way to run program is to run from main directory:

```bash
$ pip install -r /path/to/requirements.txt
$ python3 main.py
```

However it can by adjusted with many arguments

### Arguments
```python
# -f FILENAME -m MODEL -rd RESPONSE_DELAY -dd DROP_DUPLICATES -e EVALUATE -tts TEXT_TO_SPEECH -asr AUTOMATIC_SPEECH_RECOGNITION -fv FEMALE_VOICE
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
parser.add_argument('-tts', '--text_to_speech', dest='tts',action='store_true',
                    help='Use text-to-speech for system utterances')
parser.add_argument('-fv', '--female_voice', dest='female_voice',action='store_true',
                    help='Set female voice for system utterances')
parser.add_argument('-asr', '--automatic_speech_recognition', dest='asr',action='store_true',
                    help='Use automatic speech recognition (ASR) for user utterances')
```

For example to run part 1A with baseline ruled-based model and dropped duplicates and text-to-speech female voice:
```bash
$ python3 main.py -t 1A -m brb -dd -tts -fv
```

## Data
- [dialog_acts.dat](data/dialog_acts.dat): dialogs from the second Dialog State Tracking Challenge (DSTC 2, see https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/). The data consist of 3235 dialogs in the restaurant domain. Each dialog represents an interaction between a user and a system.
- [restaurant_info.csv)](data/restaurant_info.csv): available restaurants database
- [all_dialogs.txt](all_dialogs.txt): example dialogs for a system design inspiration

## Code structure
```
├──  data
│    └──  dialog_acts.dat
│    └──  restaurant_info.csv)
│    └──  all_dialogs.txt
│
│
├──  reports
│    └── eval
│        └── {datetime}_eval_report.txt   - sample generated evaluation report
│        └── {datetime}_conf_matrix.txt   - sample generated confusion matrix
│
│
├──  src
│    └──  csv_update.py                  - generates updated csv file
│    └──  evaluations.py                  - class contains all evaluation metrics
│    └──  interface.py                    - main app interface
│    └──  models.py                       - source code of the all models (baselines and ML)
│    └──  state_machine.py                - dialog state machine logic and output implementation
│    └──  utils.py                        - other functions like preparing data
│
│
├──  tmp                                  - temporary markdown notes
|
|
└──  main.py
```

## Experiments

**TBA** October '24

## References

**[1]** [Methods in AI Research, UU course](https://osiris-student.uu.nl/onderwijscatalogus/extern/cursus?cursuscode=INFOMAIR&taal=en&collegejaar=huidig)



