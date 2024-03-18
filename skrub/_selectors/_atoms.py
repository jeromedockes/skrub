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


class Filter(Selector):
    def __init__(self, predicate, on_error="raise"):
        self.predicate = predicate
        allowed = ["raise", "reject", "accept"]
        if on_error not in allowed:
            raise ValueError(f"'on_error' must be one of {allowed}. Got {on_error!r}")
        self.on_error = on_error

    def matches(self, col):
        try:
            return self.predicate(col)
        except Exception:
            if self.on_error == "raise":
                raise
            if self.on_error == "accept":
                return True
            assert self.on_error == "reject"
            return False

    def __repr__(self):
        return f"filter({self.predicate!r})"


def filter(predicate, on_error="raise"):
    return Filter(predicate, on_error=on_error)


class FilterNames(Selector):
    def __init__(self, predicate):
        self.predicate = predicate

    def matches(self, col):
        return self.predicate(sbd.name(col))

    def __repr__(self):
        return f"filter_names({self.predicate!r})"


def filter_names(predicate):
    return FilterNames(predicate)


class CreatedBy(Selector):
    def __init__(self, *transformers):
        self.transformers = transformers

    def matches(self, col):
        col_name = sbd.name(col)
        for step in self.transformers:
            if hasattr(step, "created_outputs_"):
                if col_name in step.created_outputs_:
                    return True
            elif col_name in step.get_feature_names_out():
                return True
        return False

    def __repr__(self):
        transformers_repr = f"<any of {len(self.transformers)} transformers>"
        return f"created_by({transformers_repr})"


def created_by(*transformers):
    return CreatedBy(*transformers)
