import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from typing import Dict, List


# TODO change docstring styles

def prepare_data(path: str, drop_duplicates: bool = False, vectorize: bool = False) -> tuple:
    """Preparing data for modeling - creating target, getting lowercase,
        splitting and optionally dropping duplicates.

    Parameters:
    ---
    path: str
        path to the data file in .dat format
    drop_duplicates: bool
        True is the duplicates should be dropped, empty either way

    Returns:
    ---
    tuple
        list of data split into X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path, names=['sentence'])

    df['target'] = df['sentence'].apply(lambda x: x.split()[0].lower())
    df['sentence'] = df['sentence'].apply(lambda x: x.split(' ', 1)[1].lower())

    if drop_duplicates:
        df = df.drop_duplicates(subset=['sentence'], keep='first')

    X_train, X_test, y_train, y_test = train_test_split(
        df['sentence'], df['target'], test_size=0.15, random_state=42)

    if vectorize:
        X_train, X_test, vectorizer = vectorize_data(X_train, X_test)
    else: vectorizer = None
        
    return X_train, X_test, y_train, y_test, vectorizer


def vectorize_data(X_train: np.array, X_test: np.array) -> tuple:
    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, vectorizer


def get_possible_restaurants(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_possible_choices(path: str) -> Dict[str, List]:
    """ Prepares all possible choice for each restaurant field
    """
    df_dialogues = pd.read_csv(path)

    price_range_list = df_dialogues['pricerange'].unique()
    area_list = df_dialogues['area'].unique()[:-1]
    food_list = df_dialogues['food'].unique()

    choices: dict = {
        'food': food_list,
        'area': area_list,
        'pricerange': price_range_list,
    }

    return choices
