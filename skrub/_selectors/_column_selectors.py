import fnmatch
import functools
import re

from .. import _dataframe as sbd
from ._base import column_selector, Filter

__all__ = [
    "glob",
    "regex",
    "filter_names",
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


def _glob(column, pattern):
    return fnmatch.fnmatch(sbd.name(column), pattern)

def glob(pattern):
    return Filter(_glob, args=(pattern,), name="glob")


@column_selector
def regex(column, pattern):
    return re.match(sbd.name(column), pattern) is not None


@column_selector
def filter_names(column, predicate):
    return predicate(sbd.name(column))


#
# Selectors based on data types
#


@functools.lru_cache
def numeric():
    return Filter(sbd.is_numeric, name="numeric")


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


#
# Selectors based on column values, computed statistics
#


@column_selector
def cardinality_below(column, threshold):
    try:
        return sbd.n_unique(column) < threshold
    except Exception:
        # n_unique can fail for example for polars columns with dtype Object
        return False
