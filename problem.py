# noqa: D100
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import time as time
import os, sys
import pdb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# Ramp imports
import rampwf as rw
from rampwf.score_types import BaseScoreType

# scoring
# from pysurvival.utils.metrics import integrated_brier_score, concordance_index
from sksurv.metrics import concordance_index_ipcw

data_dir = 'data'

def check_data_exist(directory):
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
    spliter = GroupShuffleSplit(n_splits=n_splits, test_size=.2, random_state=42)
    non_censored = y[:,0] # Contains the content of the 'death' column, to keep track of which patient was censored
    splits = spliter.split(X, y, non_censored)
    return splits


def _read_data(path, dir_name):
    path_to_database = os.path.join(path, data_dir, dir_name)
    path_X = os.path.join(path_to_database, 'X.csv')
    path_y = os.path.join(path_to_database, 'y.csv')
    check_data_exist(path_to_database)
    X = pd.read_csv(path_X)
    y = pd.read_csv(path_y).values
    return X, y


def get_train_data(path="."):
    """Return the train data."""
    return _read_data(path, 'train')


def get_test_data(path="."):
    """Return the test data."""
    return _read_data(path, 'test')


def _get_y_tot(path="."):
    """Return the concatenation of the train and the test labels."""
    _, y_train = get_train_data()
    _, y_test = get_test_data()
    y_tot = np.vstack((y_train, y_test))
    return y_tot


class ConcordanceIndex(BaseScoreType):
    """Concordance Index taking the censoring distribution from the training data into account."""
    def __init__(self, name='concordance_index', precision=4):
        self.name = name
        self.precision = precision
        is_lower_the_better = False
        minimum = 0.0
        maximum = 1.0
        y_tot = _get_y_tot()
        self.max_time = y_tot[1].max() # Max survival time in the whole dataset (same as in the train dataset)
        _, y_train = get_train_data()
        self.struct_y_train = self._to_structured_array(y_train) # To estimate the censoring distribution


    def _to_structured_array(self, y):
        """Create a structured array containing the event and the time to the event."""
        # y[0] = event, y[1] = time
        struct_y = y.ravel().view([('event', y[0].dtype), ('time', y[1].dtype)]).astype('bool, <i8')
        return struct_y


    def _survival_to_risk(self, y_pred):
        """Risk is a relative value, inversely correlated to life expectancy."""
        max_y = max(y_pred, self.max_time)
        return (max_y + 1) - y_pred # Return a positive risk


    def __call__(self, y_true, y_pred):
        risk = self._survival_to_risk(y_pred)
        struct_y_test = self._to_structured_array(y_test)
        score = concordance_index(self.struct_y_train, struct_y_test, risk)[0]
        return score


problem_title = 'Breast cancer survival prediction'
Predictions = rw.prediction_types.make_regression()
workflow = rw.workflows.Regressor()
score_types = [
    ConcordanceIndex(name='concordance_index')
]   
