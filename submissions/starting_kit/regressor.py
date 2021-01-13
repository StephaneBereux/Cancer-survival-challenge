from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import pdb

class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        # TODO : eliminate 'index' and choose if 'death' goes in X or in y
        preprocessor = make_column_transformer(('drop', ['index', 'death']), remainder='passthrough')
        pca = PCA(n_components=10)
        regressor = LinearRegression(n_jobs=-1)
        self.regr = Pipeline([('preprocessor', preprocessor), ('pca', pca), ('regressor', regressor)])
        return 

    def fit(self, X, y=None):
        y_to_predict = y[:,1] # We are only interested in predicting the survival time, not the censoring
        self.regr.fit(X,y_to_predict)
        return self
    
    def predict(self, X):
        print(X.shape)
        print(self.regr.predict(X).shape)
        return self.regr.predict(X)

