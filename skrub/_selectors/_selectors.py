import fnmatch
import re

from .. import _dataframe as sbd
from ._base import Filter, NameFilter

__all__ = [
    "glob",
    "regex",
    "numeric",
    "any_date",
    "categorical",
    "string",
    "boolean",
    "cardinality_below",
]

#
# Selectors based on column names
#


def glob(pattern):
    return NameFilter(fnmatch.fnmatch, args=(pattern,), name="glob")


def regex(pattern):
    return NameFilter(re.match, args=(pattern,), name="regex")


#
# Selectors based on data types
#


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


#
# Selectors based on column values, computed statistics
#


def _cardinality_below(column, threshold):
    try:
        return sbd.n_unique(column) < threshold
    except Exception:
        # n_unique can fail for example for polars columns with dtype Object
        return False


def cardinality_below(threshold):
    return Filter(_cardinality_below, args=(threshold,), name="cardinality_below")
