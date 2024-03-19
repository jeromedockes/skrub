import fnmatch
import re

from .. import _dataframe as sbd
from ._base import Selector


class Glob(Selector):
    def __init__(self, pattern):
        self.pattern = pattern

    def matches(self, col):
        return fnmatch.fnmatch(sbd.name(col), self.pattern)

    def __repr__(self):
        return f"glob({self.pattern!r})"


def glob(pattern):
    return Glob(pattern)


class Regex(Selector):
    def __init__(self, pattern):
        self.pattern = pattern

    def matches(self, col):
        return re.match(self.pattern, sbd.name(col)) is not None

    def __repr__(self):
        return f"regex({self.pattern!r})"


def regex(pattern):
    return Regex(pattern)


class FilterNames(Selector):
    def __init__(self, predicate):
        self.predicate = predicate

    def matches(self, col):
        return self.predicate(sbd.name(col))

    def __repr__(self):
        return f"filter_names({self.predicate!r})"


def filter_names(predicate):
    return FilterNames(predicate)
