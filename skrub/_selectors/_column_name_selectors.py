import fnmatch
import re

from .. import _dataframe as sbd
from ._base import column_selector


@column_selector
def glob(column, pattern):
    return fnmatch.fnmatch(sbd.name(column), pattern)


@column_selector
def regex(column, pattern):
    return re.match(sbd.name(column), pattern) is not None


@column_selector
def filter_names(column, predicate):
    return predicate(sbd.name(column))
