from sklearn.base import BaseEstimator

from . import _dataframe as sbd


class PandasConvertDTypes(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        if not sbd.is_pandas(column):
            return NotImplemented
        self.original_dtype_ = sbd.dtype(column)
        column = sbd.pandas_convert_dtypes(column)
        self.target_dtype_ = sbd.dtype(column)
        return column

    def transform(self, column):
        column = sbd.pandas_convert_dtypes(column)
        column = sbd.cast(column, self.target_dtype_)
        return column
