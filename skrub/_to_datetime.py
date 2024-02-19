from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from . import _datetime_utils
from . import _selectors as s

_SAMPLE_SIZE = 1000


class ToDatetime(BaseEstimator):
    __single_column_transformer__ = True

    def __init__(self, datetime_format=None):
        self.datetime_format = datetime_format

    def _get_datetime_format(self, column):
        if self.datetime_format is not None:
            return self.datetime_format
        not_null = sbd.drop_nulls(column)
        sample = sbd.sample(not_null, n=min(_SAMPLE_SIZE, sbd.shape(not_null)[0]))
        sample = sbd.to_pandas(sample)
        sample = sbd.pandas_convert_dtypes(sample)
        if not sbd.is_string(sample):
            return None
        if not _datetime_utils.is_column_datetime_parsable(sample):
            return None
        return _datetime_utils.guess_datetime_format(sample, random_state=0)

    def fit_transform(self, column):
        if sbd.is_anydate(column):
            self.datetime_format_ = None
            self.output_dtype_ = sbd.dtype(column)
            return column
        if not (sbd.is_string(column) or sbd.is_object(column)):
            return NotImplemented

        datetime_format = self._get_datetime_format(column)
        if datetime_format is None:
            return NotImplemented

        self.datetime_format_ = datetime_format
        sample_as_datetime = sbd.to_datetime(
            sbd.sample(column, min(10, sbd.shape(column)[0])),
            format=self.datetime_format_,
            strict=False,
        )
        self.output_dtype_ = sbd.dtype(sample_as_datetime)
        return self.transform(column)

    def transform(self, column):
        return _col_to_datetime(
            column, format=self.datetime_format_, dtype=self.output_dtype_
        )

    def fit(self, column):
        self.fit_transform(column)
        return self


@sbd.dispatch
def _col_to_datetime(column, format, dtype):
    column = sbd.to_datetime(column, format=format, strict=False)
    return sbd.cast(column, dtype)


@_col_to_datetime.specialize("polars")
def _col_to_datetime_polars(column, format, dtype):
    import polars as pl

    if format is None:
        return pl.col(column.name).cast(dtype)
    return pl.col(column.name).str.to_datetime(format=format).cast(dtype)


@sbd.dispatch
def to_datetime(df, format=None):
    """Convert DataFrame or column to Datetime dtype.

    Parameters
    ----------
    df : dataframe or column
        The dataframe or column to transform.

    format : str or None
        Format string to use to parse datetime strings.
        See the reference documentation for format codes:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes .

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import to_datetime
    >>> X = pd.DataFrame(dict(a=[1, 2], b=["01/02/2021", "21/02/2021"]))
    >>> X
       a           b
    0  1  01/02/2021
    1  2  21/02/2021
    >>> to_datetime(X)
       a          b
    0  1 2021-02-01
    1  2 2021-02-21
    """
    raise TypeError(
        "Input to skrub.to_datetime must be a pandas or polars Series or DataFrame."
        f" Got {type(df)}."
    )


@to_datetime.specialize("pandas", "DataFrame")
@to_datetime.specialize("polars", "DataFrame")
def _to_datetime_dataframe(df, format=None):
    return s.all().use(ToDatetime(datetime_format=format)).fit_transform(df)


@to_datetime.specialize("pandas", "Column")
@to_datetime.specialize("polars", "Column")
def _to_datetime_column(column, format=None):
    result = ToDatetime(datetime_format=format).fit_transform(column)
    if result is NotImplemented:
        return column
    return result
