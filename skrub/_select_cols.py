from sklearn.base import BaseEstimator, TransformerMixin

from .dataframe import get_df_namespace


def _check_columns(df, columns):
    if not hasattr(df, "columns"):
        return
    diff = set(columns) - set(df.columns)
    if not diff:
        return
    raise ValueError(
        f"The following columns were not found in the input DataFrame: {diff}"
    )


class SelectCols(TransformerMixin, BaseEstimator):
    """Select a subset of a DataFrame's columns.

    A ``ValueError`` is raised if any of the provided column names are not in
    the dataframe.

    Parameters
    ----------
    cols: list of str
        The columns to select.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 2], "B": [10, 20], "C": ["x", "y"]})
    >>> df
       A   B  C
    0  1  10  x
    1  2  20  y
    >>> SelectCols(["C", "A"]).fit_transform(df)
       C  A
    0  x  1
    1  y  2
    >>> SelectCols(["X", "A"]).fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: The following columns were not found in the input DataFrame: {'X'}

    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        _check_columns(X, self.cols)
        return self

    def transform(self, X):
        _check_columns(X, self.cols)
        namespace, _ = get_df_namespace(X)
        return namespace.select(X, self.cols)


class DropCols(TransformerMixin, BaseEstimator):
    """Drop a subset of a DataFrame's columns.

    The other columns are kept in their original order. A ``ValueError`` is
    raised if any of the provided column names are not in the dataframe.

    Parameters
    ----------
    cols: list of str
        The columns to drop.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 2], "B": [10, 20], "C": ["x", "y"]})
    >>> df
       A   B  C
    0  1  10  x
    1  2  20  y
    >>> DropCols(["A", "C"]).fit_transform(df)
        B
    0  10
    1  20
    >>> DropCols(["X"]).fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: The following columns were not found in the input DataFrame: {'X'}

    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        _check_columns(X, self.cols)
        return self

    def transform(self, X):
        _check_columns(X, self.cols)
        namespace, _ = get_df_namespace(X)
        return namespace.select(X, [c for c in X.columns if c not in self.cols])
