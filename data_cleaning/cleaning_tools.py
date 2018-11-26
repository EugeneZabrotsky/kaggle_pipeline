import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator


def get_columns_with_missing(df):
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
    return cols_with_missing


def get_quanti_quali_columns(df, columns=None):
    '''
    returns tuple of two lists of column names with
    Quantitative and qualitive features.
    '''
    if columns is None:
        _columns = df.columns
    else:
        _columns = columns.copy()

    quantitative = [f for f in _columns if df.dtypes[f] != 'object']
    qualitative = [f for f in _columns if df.dtypes[f] == 'object']
    return quantitative, qualitative


def get_constant_features(df):
    const_features = [col for col in df.columns if len(df[col].unique()) == 1]
    return const_features


def drop_constant_features(df):
    const_features = get_constant_features(df)
    return df.drop(const_features, axis=1)

# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html


class MySimpleImputer(SimpleImputer):
    '''
    Accepts DataFrame and transforms DataFrame only on specified 'columns'.
    Returns DataFrame.
    '''

    def __init__(self,
                 columns,
                 missing_values=np.nan,
                 strategy='mean',
                 fill_value=None,
                 verbose=0,
                 copy=True):
        self._columns = columns
        super().__init__(missing_values, strategy, fill_value, verbose, copy)

    def fit(self, X, columns=None):
        assert self._columns is not None
        super().fit(X[self._columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self._columns] = pd.DataFrame(
            super().transform(X_copy[self._columns]), columns=self._columns)
        return X_copy

    def fit_transform(self, X, columns=None):
        assert self._columns is not None
        return self.fit(X[self._columns]).transform(X)


class MyMissingIndicator(MissingIndicator):

    def __init__(self,
                 columns,
                 missing_values=np.nan,
                 sparse='auto',
                 error_on_new=True):
        self._columns = columns
        super().__init__(
            missing_values,
            features='all',
            sparse=sparse,
            error_on_new=error_on_new)

    def fit(self, X):
        assert self._columns is not None
        super().fit(X[self._columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        new_columns = list(
            map(lambda x: 'missing_indicator_' + x, self._columns))
        X_copy[new_columns] = pd.DataFrame(
            super().transform(X_copy[self._columns]), columns=self._columns)
        return X_copy

    def fit_transform(self, X):
        assert self._columns is not None
        return self.fit(X[self._columns]).transform(X)
