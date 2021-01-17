from sksurv.linear_model import CoxPHSurvivalAnalysis
from problem import to_structured_array

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression


def get_underexpressed_columns(X):
    """Columns with too many zeros are problematic for Cox regression : we don't take them into account."""
    expression_threshold = 100
    sum_x = X.sum()
    underexpressed_columns = X.columns[(sum_x < expression_threshold).to_numpy()]
    underexpressed_columns = [column.split('\n')[0] for column in underexpressed_columns]
    return underexpressed_columns


class FeatureFilter(BaseEstimator, TransformerMixin):
    """ A transformer that only keeps the genes listed in the feature_i."""
    def __init__(self, num_feature, to_drop_columns=[]):
        if num_feature == 1:
            self.feature_genes = columns_1
        elif num_feature == 2:
            self.feature_genes = columns_2
        elif num_feature == 3:
            self.feature_genes = columns_3
        elif num_feature == 4:
            self.feature_genes = columns_4
        else:
            print('num_feature is between 1 and 4.')
        self.feature_genes = set(self.feature_genes) - set(to_drop_columns)
        

    def fit(self, X, y=None):
        return self


    def transform(self, X):
        filtered_X = X.filter(self.feature_genes)
        return filtered_X

    

class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, num_feature, to_drop_columns=[]):
        feature_transformer = FeatureFilter(num_feature, to_drop_columns)
        scaler = StandardScaler()
        regressor = CoxPHSurvivalAnalysis()
        self.regr = Pipeline([('feature_transformer', feature_transformer), 
                              ('scaler', scaler),
                              ('regressor', regressor)])
        return 

    def fit(self, X, E_y=None):
        struct_E_y = to_structured_array(E_y)
        self.regr.fit(X, struct_E_y)
        return self
    
    def predict(self, X):
        risk_pred = self.regr.predict(X)
        y_pred = (max(risk_pred) + 1) - risk_pred
        return y_pred