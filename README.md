# Restaurant reccomendations dialog system
Project consists a design, implementation, evaluatation and reports about a restaurant recommendations dialog system using various methods from AI, such as domain modeling, text classification using machine learning and user experience testing. 

## Table of Contents

1. [Purpose of the project](#purpose-of-the-project)
2. [Requirements](#requirements)
3. [Run](#run)
4. [Data](#data)
5. [Code Structure](#code-structure)
6. [Experiments](#experiments)
7. [References](#references)

## Purpose of the project
Project made for Utrecht University course:

[Methods in AI Research](https://osiris-student.uu.nl/onderwijscatalogus/extern/cursus?cursuscode=INFOMAIR&taal=en&collegejaar=huidig)

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
The project was developed and test with ```python 3.9.12``` and libraries from [requirements.txt](requirements.txt):

```txt
- pandas=1.4.4
- scikit-learn=1.0.2
- numpy=1.23.5
- seaborn=0.11.2
- matplotlib=3.7.0
- textdistance=4.2.1
```

## Run
The easier way to run program is to run from main directory:

```bash
$ pip install -r /path/to/requirements.txt
$ python3 main.py
```

However it can by adjusted with many arguments

### Arguments
```python
# -f FILENAME -t TASK -m MODEL -dd DROP_DUPLICATES -e EVALUATE
parser.add_argument('-f', '--filename', dest='datapath',
                    default='dialog_acts.dat', help='File in data folder in .dat format')
parser.add_argument('-t', '--task', dest='task',
                    default='1B', help='Project subpart to run (1A, 1B, 1C')
parser.add_argument('-m', '--model', dest='model_name', default='fnn',
                    help="""Model to train and predict:
                            bm: BaselineMajor
                            brb: BaselineRuleBased
                            lr: LogisticRegressio
                            fnn: FeedForwardNeuralNetwork""")
parser.add_argument('-dd', '--drop_duplicates', dest='drop_duplicates', action='store_true',
                    help='Drop duplicate entries from data')
parser.add_argument('-e', '--evaluate', dest='evaluate',action='store_true',
                    help='Make evaluation and save it to file')
```

For example to run part 1A with baseline ruled-based model and dropped duplicates:
```bash
$ python3 main.py -t 1A -m brb -dd
```

## Data


