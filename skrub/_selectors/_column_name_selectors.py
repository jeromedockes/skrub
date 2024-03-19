import fnmatch
import re

from .. import _dataframe as sbd
from ._base import Filter


def glob(pattern):
    return Filter(
        lambda c: fnmatch.fnmatch(sbd.name(c), pattern), name="glob", args=(pattern,)
    )


def regex(pattern):
    compiled = re.compile(pattern)
    return Filter(
        lambda c: compiled.match(sbd.name(c)) is not None,
        name="regex",
        args=(pattern,),
    )


def filter_names(predicate):
    return Filter(
        lambda c: predicate(sbd.name(c)), name="filter_names", args=(predicate,)
    )
