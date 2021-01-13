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
from rampwf.workflows import SKLearnPipeline
from rampwf.utils.importing import import_module_from_source
from rampwf.prediction_types.base import BasePrediction

# scoring
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


def _survival_regression_init(self, y_pred=None, y_true=None, n_samples=None):
    """The participants only predict the survival time in y_pred, while y_true 
    also contains the censoring data along the first coordinate. Thus :
    y_pred.shape = (n_samples, n_columns) and y_true.shape = (n_samples, 1 + n_columns). 
    This prediction type handles it."""
    if y_pred is not None:
        self.y_pred = np.array(y_pred)
    elif y_true is not None:
        self.y_pred = np.array(y_true[:,1])
    elif n_samples is not None:
        if self.n_columns == 0:
            shape = (n_samples)
        else:
            shape = (n_samples, self.n_columns)
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')
    self.check_y_pred_dimensions()


def make_survival_regression(label_names=[], index_censoring_data=0):
    """index_censoring_data (int) : index of the columns of y_true 
        which contains the censoring data. Default is 0."""
    Predictions = type(
        '_SurvivalRegression',
        (BasePrediction,),
        {'label_names': label_names,
         'n_columns': len(label_names),
         'n_columns_true': len(label_names),
         '__init__': _survival_regression_init,
         })
    return Predictions


class Regressor_df(object):
    """Regressor workflow allowing X to be a dataframe."""

    def __init__(self, workflow_element_names=['regressor']):
        super().__init__()
        self.element_names = workflow_element_names

        
    def train_submission(self, module_path, X_df, y, train_is=None):
        try:
            if train_is is None:
                train_is = slice(None, None, None)
            regressor = import_module_from_source(
                os.path.join(module_path, self.element_names[0] + '.py'),
                self.element_names[0],
                sanitize=True
            )
            reg = regressor.Regressor()
            reg.fit(X_df.iloc[train_is], y[train_is])
            print('fitting done')
        except:
            print('fit')
            pdb.set_trace()
        return reg


    def test_submission(self, trained_model, X_df):
        try:
            print('here')
            reg = trained_model
            y_pred = reg.predict(X_df)
        except:
            print('pred')
            pdb.set_trace()
        return y_pred


def make_regressor_df_worflow():
    """Define new workflow, similar to Regressor but where X is a Dataframe."""
    return Regressor_df()


class ConcordanceIndex(BaseScoreType):
    """Concordance Index taking the censoring distribution from the training data into account."""
    def __init__(self, name='concordance_index', precision=4):
        self.name = name
        self.precision = precision
        is_lower_the_better = False
        minimum = 0.0
        maximum = 1.0
        y_tot = _get_y_tot()
        self.max_time = y_tot[:,1].max() # Max survival time in the whole dataset (same as in the train dataset)
        _, y_train = get_train_data()
        self.struct_y_train = self._to_structured_array(y_train) # To estimate the censoring distribution


    def _to_structured_array(self, y):
        """Create a structured array containing the event and the time to the event."""
        # y[0] = event, y[1] = time
        struct_y = y.ravel().view([('event', y[0][0].dtype), ('time', y[0][1].dtype)]).astype('bool, <i8')
        return struct_y


    def _survival_to_risk(self, y_pred):
        """Risk is a relative value, inversely correlated to life expectancy."""
        max_y = max(y_pred.max(), self.max_time)
        return (max_y + 1) - y_pred # Return a positive risk


    def __call__(self, y_true, y_pred):
        pdb.set_trace()
        risk = self._survival_to_risk(y_pred)
        struct_y_test = self._to_structured_array(y_true)
        score = concordance_index(self.struct_y_train, struct_y_test, risk)[0]
        return score


problem_title = 'Breast cancer survival prediction'
Predictions = make_survival_regression()
workflow = make_regressor_df_worflow()
score_types = [
    ConcordanceIndex(name='concordance_index')
]   
