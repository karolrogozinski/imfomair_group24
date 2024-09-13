import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def prepare_data(path: str, drop_duplicates: bool = False) -> tuple:
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
        
    return X_train, X_test, y_train, y_test

# TODO implement Bag of Words
# BoW implementation through sklearn CountVectorizer. We can personalize the implementation if we have to.
def prepare_data_bow(path: str, drop_duplicates: bool = False) -> tuple:
        """
        Here, we prepare bag of words data in a different function so that 
            we can keep using the old one without any problem in baseline models.
            This creates some repeated code but I did not want to change anything 
            already implemented for baseline models.

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

        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
            df['sentence'], df['target'], test_size=0.15, random_state=42)

        # BoW for the training data
        # We are not using an explicit tokenizer since CountVectorizer uses a default one.
        vectorizer = CountVectorizer()

        X_train = vectorizer.fit_transform(X_train_pre)
        X_test = vectorizer.transform(X_test_pre)

        # Got to return the vectorizer too, because we need to convert console input sentences to BoW as well.
        return X_train, X_test, y_train_pre, y_test_pre, vectorizer
        