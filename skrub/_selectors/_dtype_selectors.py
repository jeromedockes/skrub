from .. import _dataframe as sbd
from ._base import Filter


def numeric():
    return Filter(sbd.is_numeric, name="numeric")


def any_date():
    return Filter(sbd.is_any_date, name="any_date")


def categorical():
    return Filter(sbd.is_categorical, name="categorical")


def string():
    return Filter(sbd.is_string, name="string")


def boolean():
    return Filter(sbd.is_bool, name="boolean")
