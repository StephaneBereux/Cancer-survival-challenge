from sksurv.linear_model import CoxPHSurvivalAnalysis
from problem import to_structured_array

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression


def get_unexpressed_columns(X_train):
    mean_x = X_train.mean()
    unexpressed_columns = X_train.columns[(mean_x == 0.).to_numpy()]
    unexpressed_columns = [column.split('\n')[0] for column in unexpressed_columns]
    return unexpressed_columns
    
    
class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pca = PCA(n_components=10)
        regressor = CoxPHSurvivalAnalysis()
        self.regr = Pipeline([('pca', pca), ('regressor', regressor)])
        return 

    def fit(self, X, E_y=None):
        self.to_drop_columns = get_unexpressed_columns(X)
        X.drop(columns=self.to_drop_columns, inplace=True)
        struct_E_y = to_structured_array(E_y)
        self.regr.fit(X, struct_E_y)
        return self
    
    def predict(self, X):
        X.drop(columns=self.to_drop_columns, inplace=True)
        risk_pred = self.regr.predict(X)
        y_pred = (max(risk_pred) + 1) - risk_pred
        return y_pred
