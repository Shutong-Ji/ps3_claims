import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, upper_quantile, lower_quantile):
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile

    def fit(self, X, y=None):
        self.upper_quantile_ = np.quantile(X, self.upper_quantile * 100)
        self.lower_quantile_ = np.quantile(X, self.lower_quantile * 100)
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        return X_clipped
