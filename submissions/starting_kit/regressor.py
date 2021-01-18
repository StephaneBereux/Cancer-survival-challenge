from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
import pandas as pd 
from scipy.stats import pearsonr
import math as math
from problem import to_structured_array

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def compute_correlations(X, y_log):
    """Compute the correlation and the associated p-values between the genes and the survival times."""
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    gene_index, corr_r, p_values = [], [], []
    corr_df = pd.DataFrame(columns = {'genes', 'correlation', 'p_values'})
    for i, gene in enumerate(scaled_X.T):
        r, p_val = pearsonr(gene, log_time)
        if not math.isnan(r): # If the correlation is valid
            gene_index.append(X.columns[i])
            corr_r.append(r)
            p_values.append(p_val)
    corr_df = pd.DataFrame({'correlation' : corr_r, 
            'p_value' : p_values} , index = gene_index) 
    return corr_df


def get_significant_genes(X, E_y):
    """Get the genes with the correlation the most statistically reliable."""
    E = E_y[:,0]
    y = E_y[:,1]
    log_time = np.log(y)
    correlation_df = compute_correlations(X, log_time)
    display_correlations(correlation_df)

    # we use Bonferroni correction
    bonferroni_alpha = 0.05 / X.shape[1]
    sign_corr = correlation_df[np.abs(correlation_df['p_value']) < bonferroni_alpha]
    sign_correlation_genes = sign_corr.index.to_numpy()
    genes_of_interest = X.filter(sign_correlation_genes)
    return genes_of_interest.columns.to_numpy()


def get_underexpressed_columns(X):
    """Columns with too many zeros are problematic for Cox regression : we don't take them into account."""
    expression_threshold = 100
    sum_x = X.sum()
    underexpressed_columns = X.columns[(sum_x < expression_threshold).to_numpy()]
    underexpressed_columns = [column.split('\n')[0] for column in underexpressed_columns]
    return underexpressed_columns


class GenesFilter(BaseEstimator, TransformerMixin):
    """ A transformer that only keeps the significant genes from the CENSUS database."""
    def __init__(self):
        return
    

    def fit(self, X, y=None):
        self.to_drop_columns = get_underexpressed_columns(X)
        self.feature_genes = get_significant_genes(X,y)
        self.feature_genes = set(self.feature_genes) - set(self.to_drop_columns)
        return self


    def transform(self, X):
        filtered_X = X.filter(self.feature_genes)
        return filtered_X

    

class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        genes_filter = GenesFilter()
        scaler = StandardScaler()
        regressor = CoxPHSurvivalAnalysis()
        self.regr = Pipeline([('genes_filter', genes_filter), 
                              ('scaler', scaler),
                              ('regressor', regressor)])
        return 

    def fit(self, X, E_y=None):
        struct_E_y = to_structured_array(E_y)
        self.regr.fit(X, struct_E_y)
        return self
    
    def predict(self, X):
        risk_pred = self.regr.predict(X)
        # transform a risk into a survival time
        y_pred = (max(risk_pred) + 1) - risk_pred
        return y_pred