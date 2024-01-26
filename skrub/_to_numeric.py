from sklearn.base import BaseEstimator

from . import _dataframe as sbd


class ToNumeric(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        if sbd.is_anydate(column) or sbd.is_categorical(column):
            raise NotImplementedError()
        try:
            numeric = sbd.to_numeric(column)
            self.output_native_dtype_ = sbd.native_dtype(numeric)
            return numeric
        except Exception:
            raise NotImplementedError()

    def transform(self, column):
        return sbd.to_numeric(column, dtype=self.output_native_dtype_)
