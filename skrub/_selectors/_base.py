import functools
import inspect
from typing import Any

from .. import _dataframe as sbd
from .._add_estimator_methods import add_estimator_methods
from .._dispatch import dispatch
from .._fluent_classes import fluent_class
from .._utils import repr_args


def all():
    return All()


def cols(*columns):
    return Cols(columns)


def inv(obj):
    return ~make_selector(obj)


def make_selector(obj):
    if isinstance(obj, Selector):
        return obj
    if isinstance(obj, str):
        return cols(obj)
    if not hasattr(obj, "__iter__"):
        raise ValueError(f"selector not understood: {obj}")
    return cols(*obj)


@dispatch
def _select_col_names(df, selector):
    raise NotImplementedError()


@_select_col_names.specialize("pandas")
def _select_col_names_pandas(df, col_names):
    return df[col_names]


@_select_col_names.specialize("polars")
def _select_col_names_polars(df, col_names):
    return df.select(col_names)


def select(df, selector):
    return _select_col_names(df, make_selector(selector).expand(df))


@add_estimator_methods
class Selector:
    def matches(self, col):
        raise NotImplementedError()

    def expand(self, df):
        matching_col_names = []
        for col_name in sbd.column_names(df):
            col = sbd.col(df, col_name)
            if self.matches(col):
                matching_col_names.append(col_name)
        return matching_col_names

    def __invert__(self):
        return Inv(self)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return Or(other, self)

    def __and__(self, other):
        return And(self, other)

    def __rand__(self, other):
        return And(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __xor__(self, other):
        return XOr(self, other)

    def __rxor__(self, other):
        return XOr(other, self)

    def make_transformer(
        self,
        transformer,
        keep_original=False,
        rename_columns="{}",
        n_jobs=None,
        columnwise="auto",
    ):
        from .._on_column_selection import OnColumnSelection
        from .._on_each_column import OnEachColumn

        if isinstance(columnwise, str) and columnwise == "auto":
            columnwise = hasattr(transformer, "__single_column_transformer__")

        if columnwise:
            return OnEachColumn(
                transformer,
                keep_original=keep_original,
                rename_columns=rename_columns,
                cols=self,
                n_jobs=n_jobs,
            )
        return OnColumnSelection(
            transformer,
            keep_original=keep_original,
            rename_columns=rename_columns,
            cols=self,
        )

    def use(self, estimator):
        return PipeStep(cols=self, estimator=estimator)


@fluent_class
class PipeStep:
    cols_: Selector
    estimator_: Any
    name_: str | None = None
    keep_original_: bool = False
    rename_columns_: str = "{}"

    def _make_transformer(self, estimator=None, n_jobs=1):
        if estimator is None:
            estimator = self.estimator_
        return self.cols_.make_transformer(
            estimator,
            keep_original=self.keep_original_,
            rename_columns=self.rename_columns_,
            n_jobs=n_jobs,
        )


class All(Selector):
    def matches(self, col):
        return True

    def __repr__(self):
        return "all()"


def _check_string_list(columns):
    columns = list(columns)
    for c in columns:
        if not isinstance(c, str):
            raise ValueError(
                "Column name selector should be initialized with a list of str. Found"
                f" non-string element: {c!r}."
            )
    return columns


class Cols(Selector):
    def __init__(self, columns):
        self.columns = _check_string_list(columns)

    def matches(self, col):
        return sbd.name(col) in self.columns

    def expand(self, df):
        missing = set(self.columns).difference(sbd.column_names(df))
        if missing:
            raise ValueError(
                "The following columns are requested for selection but "
                f"missing from dataframe: {list(missing)}"
            )
        return self.columns

    def __repr__(self):
        return f"cols({', '.join(map(repr, self.columns))})"


class Inv(Selector):
    def __init__(self, complement):
        self.complement = make_selector(complement)

    def matches(self, col):
        return not self.complement.matches(col)

    def __repr__(self):
        return f"(~{self.complement!r})"


class Or(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def matches(self, col):
        return self.left.matches(col) or self.right.matches(col)

    def __repr__(self):
        return f"({self.left!r} | {self.right!r})"


class And(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def matches(self, col):
        return self.left.matches(col) and self.right.matches(col)

    def __repr__(self):
        return f"({self.left!r} & {self.right!r})"


class Sub(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def matches(self, col):
        return self.left.matches(col) and (not self.right.matches(col))

    def __repr__(self):
        return f"({self.left!r} - {self.right!r})"


class XOr(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def matches(self, col):
        return self.left.matches(col) ^ self.right.matches(col)

    def __repr__(self):
        return f"({self.left!r} ^ {self.right!r})"


class Filter(Selector):
    def __init__(self, predicate, repr_return_value=None):
        self.predicate = predicate
        self._repr_return_value = repr_return_value

    def matches(self, col):
        return self.predicate(col)

    def __repr__(self):
        if self._repr_return_value is None:
            return f"filter({self.predicate!r})"
        return self._repr_return_value


def filter(predicate):
    return Filter(predicate)


def column_selector(f):
    @functools.wraps(f)
    def make_filter(*args, **kwargs):
        repr_ = f"{f.__name__}({repr_args(args, kwargs)})"
        return Filter(lambda col: f(col, *args, **kwargs), repr_return_value=repr_)

    sig = inspect.signature(f)
    make_filter.__signature__ = sig.replace(
        parameters=list(sig.parameters.values())[1:], return_annotation=Selector
    )
    return make_filter
