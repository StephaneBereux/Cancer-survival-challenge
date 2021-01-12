# noqa: D100
import numpy as np
import pandas as pd
import time as time
import os, sys

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# Ramp imports
import rampwf as rw
from rampwf.score_types import BaseScoreType

# for gene names converting
import mygene as mg

# for data fetching
import xenaPython as xena

problem_title = 'Breast cancer survival prediction'
data_dir = 'data'
Predictions = rw.prediction_types.make_regression()


def check_data_exist():
    """Exit if the data weren't previously downloaded."""
    if not os.path.exists(directory):
        print('Data are missing. Maybe you should first run \n " python download_data.py "')
        sys.exit()


def get_cv(X, y):
    """Create cross-validation for the ramp test."""

    test = os.getenv('RAMP_TEST_MODE', 0)
    n_splits = 4
    if test:
        n_splits = 2
    spliter = GroupShuffleSplit(n_splits=n_splits, test_size=.2,
                            random_state=42)
    non_censored = X['death']
    splits = spliter.split(X, y, non_censored)

    # take only 500 samples per test subject for speed
    def limit_test_size(splits):
        rng = np.random.RandomState(42)
        for train, test in splits:
            yield (train, test[rng.permutation(len(test))[:500]])
    return limit_test_size(splits)


def _read_data(path, dir_name):
    path_to_database = os.path.join(path, data_dir, dir_name)
    path_X = os.path.join(path_to_database, 'X.csv')
    path_E = os.path.join(path_to_database, 'E.csv')
    path_y = os.path.join(path_to_database, 'y.csv')
    check_data_exist(path_to_database)
    X = pd.read_csv(path_to_database)
    y = df.pop('time')
    return X, E, y


def get_train_data(path="."):
    """Return the train data."""
    return _read_data(path, 'train')


def get_test_data(path="."):
    """Return the test data."""
    return _read_data(path, 'test')

class ConcordanceIndex(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='concordance_index', precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        mask = ~np.any(np.isnan(y_proba), axis=1)

        score = concordance_index(y_true_proba[mask],
                                  y_proba[mask],
                                  average='samples')
        return score


score_types = [
    ConcordanceIndex(name='concordance_index'),
    IntegratedBrierScore(name='integrated_brier_score')
]   
