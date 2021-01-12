from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pca = PCA(n_components=10)
        regressor = LinearRegression(n_jobs=-1)
        self.regr = Pipeline([('pca', pca), ('regressor', regressor)])
        return self

    def fit(self, X, y=None):
        self.regr.fit(X)
        return self
    
    def predict(self, X):
        return self.regr.predict(X)

