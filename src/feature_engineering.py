import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# Custom Transformer for Aggregate Features
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'Value': ['sum', 'mean', 'std']
        })
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]
        agg_df.reset_index(inplace=True)
        return agg_df

# Custom Transformer for Extracting Date Features
class DateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X['Transaction_Hour'] = X[self.date_col].dt.hour
        X['Transaction_Day'] = X[self.date_col].dt.day
        X['Transaction_Month'] = X[self.date_col].dt.month
        X['Transaction_Year'] = X[self.date_col].dt.year
        return X