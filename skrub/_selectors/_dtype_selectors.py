from .. import _dataframe as sbd
from ._base import column_selector


@column_selector
def numeric(column):
    return sbd.is_numeric(column)


@column_selector
def any_date(column):
    return sbd.is_any_date(column)


@column_selector
def categorical(column):
    return sbd.is_categorical(column)


@column_selector
def string(column):
    return sbd.is_string(column)


@column_selector
def boolean(column):
    return sbd.is_bool(column)
