import dis
import enum
import functools
import html
import inspect
import itertools
import operator
import pathlib
import pickle
import textwrap
import traceback
import types
import typing

from sklearn.base import BaseEstimator

from .. import _dataframe as sbd
from .. import _selectors as s
from .._check_input import cast_column_names_to_strings
from .._reporting._utils import strip_xml_declaration
from .._select_cols import DropCols, SelectCols
from .._utils import short_repr
from .._wrap_transformer import wrap_transformer
from ._choosing import Choice, unwrap_chosen_or_default
from ._utils import FITTED_PREDICTOR_METHODS, _CloudPickle, attribute_error

__all__ = ["var", "X", "y", "as_expr", "deferred", "deferred_optional", "if_else"]

# Explicitly excluded from getattr because they break either pickling or the
# repl autocompletion
_EXCLUDED_STANDARD_ATTR = [
    "__setstate__",
    "__getstate__",
    "__wrapped__",
    "_partialmethod",
    "__name__",
    "__code__",
    "__defaults__",
    "__kwdefaults__",
    "__annotations__",
]

_EXCLUDED_JUPYTER_ATTR = [
    "_repr_pretty_",
    "_repr_svg_",
    "_repr_png_",
    "_repr_jpeg_",
    "_repr_javascript_",
    "_repr_markdown_",
    "_repr_latex_",
    "_repr_pdf_",
    "_repr_json_",
    "_ipython_display_",
    "_repr_mimebundle_",
    "_ipython_canary_method_should_not_exist_",
    "__custom_documentations__",
]

_EXCLUDED_PANDAS_ATTR = [
    # used internally by pandas to check an argument is actually a dataframe.
    # by raising an attributeerror when it is accessed we fail early when an
    # expression is used where a DataFrame is expected eg
    # pd.DataFrame(...).merge(skrub.X(), ...)
    #
    # polars already fails with a good error message in that situation so it
    # doesn't need special handling for polars dataframes.
    "_typ",
]

# TODO: compare with
# https://github.com/GrahamDumpleton/wrapt/blob/develop/src/wrapt/wrappers.py#L70
# and see which methods we are missing

_BIN_OPS = [
    "__add__",
    "__and__",
    "__concat__",
    "__eq__",
    "__floordiv__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lshift__",
    "__lt__",
    "__matmul__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__pow__",
    "__rshift__",
    "__sub__",
    "__truediv__",
    "__xor__",
    "__and__",
    "__or__",
]

_UNARY_OPS = [
    "__abs__",
    "__all__",
    "__concat__",
    "__inv__",
    "__invert__",
    "__not__",
    "__neg__",
    "__pos__",
]

_BUILTIN_SEQ = (list, tuple, set, frozenset)

_BUILTIN_MAP = (dict,)


class _Constants(enum.Enum):
    NO_VALUE = enum.auto()


class UninitializedVariable(KeyError):
    """
    Evaluating an expression and a value has not been provided for one of the variables.
    """


def _remove_shell_frames(stack):
    shells = [
        (pathlib.Path("IPython", "core", "interactiveshell.py"), "run_code"),
        (pathlib.Path("IPython", "utils", "py3compat.py"), "execfile"),
        (pathlib.Path("sphinx", "config.py"), "eval_config_file"),
        ("code.py", "runcode"),
    ]
    for i, f in enumerate(stack):
        for file_path, func_name in shells:
            if pathlib.Path(f.filename).match(file_path) and f.name == func_name:
                return stack[i + 1 :]
    return stack


def _format_expr_creation_stack():
    # TODO use inspect.stack() instead of traceback.extract_stack() for more
    # context lines + within-line position of the instruction (dis.Positions
    # was only added in 3.11, though)

    stack = traceback.extract_stack()
    stack = _remove_shell_frames(stack)
    fpath = pathlib.Path(__file__).parent
    stack = itertools.takewhile(
        lambda f: not pathlib.Path(f.filename).is_relative_to(fpath), stack
    )
    return traceback.format_list(stack)


class ExprImpl:
    def __init_subclass__(cls):
        params = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in cls._fields
        ]
        sig = inspect.Signature(params)

        def __init__(self, *args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            self.__dict__.update(bound.arguments)
            self.results = {}
            self.errors = {}
            try:
                self._creation_stack_lines = _format_expr_creation_stack()
            except Exception:
                self._creation_stack_lines = None
            self.is_X = False
            self.is_y = False
            if "name" not in self.__dict__:
                self.name = None
            self.description = None

        cls.__init__ = __init__

    def creation_stack_description(self):
        if self._creation_stack_lines is None:
            return ""
        return "".join(self._creation_stack_lines)

    def creation_stack_last_line(self):
        if not self._creation_stack_lines:
            return ""
        line = self._creation_stack_lines[-1]
        return textwrap.indent(line, "    ").rstrip("\n")

    def preview_if_available(self):
        return self.results.get("preview", _Constants.NO_VALUE)

    def supports_modes(self):
        return ["preview", "fit_transform", "transform"]

    def fields_required_for_eval(self, mode):
        return self._fields

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


def _check_can_be_pickled(obj):
    try:
        dumped = pickle.dumps(obj)
        pickle.loads(dumped)
    except Exception as e:
        msg = "The check to verify that the pipeline can be serialized failed."
        if "recursion" in str(e).lower():
            msg = (
                f"{msg} Is a step in the pipeline holding a reference to "
                "the full pipeline itself? For example a global variable "
                "in a `@skrub.deferred` function?"
            )
        raise pickle.PicklingError(msg) from e


def _find_dataframe(expr, func_name):
    # If a dataframe is found in an expression that is likely a mistake.
    # Eg skrub.X().join(actual_df, ...) instead of skrub.X().join(skrub.var('Z'), ...)
    from ._evaluation import find_arg

    df = find_arg(expr, lambda o: sbd.is_dataframe(o))
    if df is not None:
        return {
            "message": (
                f"You passed an actual DataFrame (shown below) to `{func_name}`. "
                "Did you mean to pass a skrub expression instead? "
                "Note: if you did intend to pass a DataFrame you can wrap it "
                "with `skrub.as_expr(df)` to avoid this error. "
                f"Here is the dataframe:\n{df}"
            )
        }
    return None


def _check_expr(f):
    """Check an expression and evaluate the preview.

    We decorate the functions that create expressions rather than do it in
    ``__init__`` to make tracebacks as short as possible: the second frame in
    the stack trace is the one in user code that created the problematic
    expression. If the check was done in ``__init__`` it might be buried
    several calls deep, making it harder to understand those errors.
    """

    @functools.wraps(f)
    def _checked_call(*args, **kwargs):
        from ._evaluation import evaluate, find_conflicts

        expr = f(*args, **kwargs)

        try:
            func_name = expr._skrub_impl.pretty_repr()
        except Exception:
            func_name = f"{f.__name__}()"

        conflicts = find_conflicts(expr)
        if conflicts is not None:
            raise ValueError(conflicts["message"])
        if (found_df := _find_dataframe(expr, func_name)) is not None:
            raise TypeError(found_df["message"])

        # Note: if checking pickling for every step is expensive we could also
        # do it in `get_estimator()` only, ie before any cross-val or
        # grid-search. or we could have some more elaborate check (possibly
        # with false negatives) where we pickle nodes separately and we only
        # check new nodes that haven't yet been checked.
        _check_can_be_pickled(expr)
        try:
            evaluate(expr, mode="preview", environment=None)
        except UninitializedVariable:
            pass
        except Exception as e:
            msg = "\n".join(traceback.format_exception_only(e)).rstrip("\n")
            raise RuntimeError(
                f"Evaluation of {func_name!r} failed.\n"
                f"You can see the full traceback above. The error message was:\n{msg}"
            ) from e

        return expr

    return _checked_call


def _get_preview(obj):
    if isinstance(obj, Expr) and "preview" in obj._skrub_impl.results:
        return obj._skrub_impl.results["preview"]
    return obj


def _check_call(f):
    @functools.wraps(f)
    def _check_call_return_value(*args, **kwargs):
        expr = f(*args, **kwargs)
        if "preview" not in expr._skrub_impl.results:
            return expr
        result = expr._skrub_impl.results["preview"]
        if result is not None:
            return expr
        try:
            func_name = expr._skrub_impl.pretty_repr()
        except Exception:
            func_name = expr._skrub_impl.get_func_name()
        msg = (
            f"Calling {func_name!r} returned None. "
            "To enable chaining steps in a pipeline, do not use functions "
            "that modify objects in-place but rather functions that leave "
            "their argument unchanged and return a new object."
        )
        raise TypeError(msg)

    return _check_call_return_value


class Expr:
    def __init__(self, impl):
        self._skrub_impl = impl

    def __sklearn_clone__(self):
        from ._evaluation import clone

        return clone(self)

    @_check_expr
    def __getattr__(self, name):
        if name in [
            "_skrub_impl",
            "get_params",
            *_EXCLUDED_STANDARD_ATTR,
            *_EXCLUDED_JUPYTER_ATTR,
            *_EXCLUDED_PANDAS_ATTR,
        ]:
            attribute_error(self, name)
        # besides the explicitly excluded attributes, returning a GetAttr for
        # any special method is unlikely to do what we want.
        if name.startswith("__") and name.endswith("__"):
            attribute_error(self, name)
        return Expr(GetAttr(self, name))

    @_check_expr
    def __getitem__(self, key):
        return Expr(GetItem(self, key))

    @_check_call
    @_check_expr
    def __call__(self, *args, **kwargs):
        impl = self._skrub_impl
        if isinstance(impl, GetAttr):
            return Expr(CallMethod(impl.parent, impl.attr_name, args, kwargs))
        return Expr(
            Call(self, args, kwargs, globals={}, closure=(), defaults=(), kwdefaults={})
        )

    @_check_expr
    def __len__(self):
        return Expr(GetAttr(self, "__len__"))()

    def __dir__(self):
        names = ["skb"]
        preview = self._skrub_impl.preview_if_available()
        if preview is not _Constants.NO_VALUE:
            names.extend(dir(preview))
        return names

    def _ipython_key_completions_(self):
        preview = self._skrub_impl.preview_if_available()
        if preview is _Constants.NO_VALUE:
            return []
        try:
            return preview._ipython_key_completions_()
        except AttributeError:
            pass
        try:
            return list(preview.keys())
        except Exception:
            pass
        return []

    @property
    def __signature__(self):
        preview = self._skrub_impl.preview_if_available()
        if callable(preview):
            return inspect.signature(preview)
        attribute_error(self, "__signature__")

    @property
    def __doc__(self):
        preview = self._skrub_impl.preview_if_available()
        if preview is _Constants.NO_VALUE:
            attribute_error(self, "__doc__")
        doc = getattr(preview, "__doc__", None)
        if doc is None:
            attribute_error(self, "__doc__")
        return f"""Skrub expression.\nDocstring of the preview:\n{doc}"""

    def __setitem__(self, key, value):
        msg = (
            "Do not modify an expression in-place. "
            "Instead, use a function that returns a new value."
            "This is necessary to allow chaining "
            "several steps in a sequence of transformations."
        )
        obj = self._skrub_impl.results.get("preview", None)
        if sbd.is_pandas(obj) and sbd.is_dataframe(obj):
            msg += (
                "\nFor example if df is a pandas DataFrame:\n"
                "df = df.assign(new_col=...) instead of df['new_col'] = ... "
            )
        raise TypeError(msg)

    def __setattr__(self, name, value):
        if name == "_skrub_impl":
            return super().__setattr__(name, value)
        raise TypeError(
            "Do not modify an expression in-place. "
            "Instead, use a function that returns a new value."
            "This is necessary to allow chaining "
            "several steps in a sequence of transformations."
        )

    def __bool__(self):
        raise TypeError(
            "This object is an expression that will be evaluated later, "
            "when your pipeline runs. So it is not possible to eagerly "
            "use its Boolean value now."
        )

    def __iter__(self):
        raise TypeError(
            "This object is an expression that will be evaluated later, "
            "when your pipeline runs. So it is not possible to eagerly "
            "iterate over it now."
        )

    def __contains__(self, item):
        raise TypeError(
            "This object is an expression that will be evaluated later, "
            "when your pipeline runs. So it is not possible to eagerly "
            "perform membership tests now."
        )

    @property
    def skb(self):
        if isinstance(self._skrub_impl, Apply):
            return ApplyNamespace(self)
        return SkrubNamespace(self)

    def __repr__(self):
        result = repr(self._skrub_impl)
        preview = self._skrub_impl.preview_if_available()
        if preview is _Constants.NO_VALUE:
            return result
        return f"{result}\nResult:\n―――――――\n{preview!r}"

    def __skrub_short_repr__(self):
        return repr(self._skrub_impl)

    def __format__(self, format_spec):
        if format_spec == "":
            return self.__skrub_short_repr__()
        if format_spec == "preview":
            return repr(self)
        raise ValueError(
            f"Invalid format specifier {format_spec!r} "
            f"for object of type {self.__class__.__name__!r}"
        )

    def _repr_html_(self):
        from ._inspection import node_report

        graph = self.skb.draw_graph().decode("utf-8")
        graph = strip_xml_declaration(graph)
        if self._skrub_impl.preview_if_available() is _Constants.NO_VALUE:
            return f"<div>{graph}</div>"
        if (name := self._skrub_impl.name) is not None:
            name_line = (
                f"<strong><samp>Name: {html.escape(repr(name))}</samp></strong><br />\n"
            )
        else:
            name_line = ""
        title = f"<strong><samp>{html.escape(short_repr(self))}</samp></strong><br />\n"
        summary = "<samp>Show graph</samp>"
        prefix = (
            f"{title}{name_line}"
            f"<details>\n<summary style='cursor: pointer;'>{summary}</summary>\n"
            f"{graph}<br /><br />\n</details>\n"
            "<strong><samp>Result:</samp></strong>"
        )
        report = node_report(self)
        if hasattr(report, "_repr_html_"):
            report = report._repr_html_()
        return f"<div>\n{prefix}\n{report}\n</div>"


def _make_bin_op(op_name):
    def op(self, right):
        return Expr(BinOp(self, right, getattr(operator, op_name)))

    op.__name__ = op_name
    return _check_expr(op)


for op_name in _BIN_OPS:
    setattr(Expr, op_name, _make_bin_op(op_name))


def _make_r_bin_op(op_name):
    def op(self, left):
        return Expr(BinOp(left, self, getattr(operator, op_name)))

    op.__name__ = f"__r{op_name.strip('_')}__"
    return _check_expr(op)


for op_name in _BIN_OPS:
    rop_name = f"__r{op_name.strip('_')}__"
    setattr(Expr, rop_name, _make_r_bin_op(op_name))


def _make_unary_op(op_name):
    def op(self):
        return Expr(UnaryOp(self, getattr(operator, op_name)))

    op.__name__ = op_name
    return _check_expr(op)


for op_name in _UNARY_OPS:
    setattr(Expr, op_name, _make_unary_op(op_name))


def _check_wrap_params(cols, how, allow_reject, reason):
    msg = None
    if not isinstance(cols, type(s.all())):
        msg = f"`cols` must be `all()` (the default) when {reason}"
    elif how not in ["auto", "full_frame"]:
        msg = f"`how` must be 'auto' (the default) or 'full_frame' when {reason}"
    elif allow_reject:
        msg = f"`allow_reject` must be False (the default) when {reason}"
    if msg is not None:
        raise ValueError(msg)


def _wrap_estimator(estimator, cols, how, allow_reject, X):
    def _check(reason):
        _check_wrap_params(cols, how, allow_reject, reason)

    if estimator in [None, "passthrough"]:
        estimator = _PassThrough()
    if isinstance(estimator, Choice):
        return estimator.map_values(
            lambda v: _wrap_estimator(v, cols, how=how, allow_reject=allow_reject, X=X)
        )
    if how == "full_frame":
        _check("`how` is 'full_frame'")
        return estimator
    if not hasattr(estimator, "transform"):
        _check("`estimator` is a predictor (not a transformer)")
        return estimator
    if not sbd.is_dataframe(X):
        _check("the input is not a DataFrame")
        return estimator
    columnwise = {"auto": "auto", "columnwise": True, "sub_frame": False}[how]
    return wrap_transformer(
        estimator, cols, allow_reject=allow_reject, columnwise=columnwise
    )


def _expr_values_provided(expr, environment):
    from ._evaluation import nodes

    all_nodes = nodes(expr)
    names = {node._skrub_impl.name for node in all_nodes}
    names.discard(None)
    intersection = names.intersection(environment.keys())
    return bool(intersection)


class SkrubNamespace:
    """The expressions' ``.skb`` attribute."""

    def __init__(self, expr):
        self._expr = expr

    def _apply(
        self,
        estimator,
        y=None,
        cols=s.all(),
        how="auto",
        allow_reject=False,
    ):
        expr = Expr(
            Apply(
                estimator=estimator,
                cols=cols,
                X=self._expr,
                y=y,
                how=how,
                allow_reject=allow_reject,
            )
        )
        return expr

    @_check_expr
    def apply(
        self,
        estimator,
        *,
        y=None,
        cols=s.all(),
        how="auto",
        allow_reject=False,
    ):
        """
        Apply a scikit-learn estimator to a dataframe or numpy array.

        Parameters
        ----------
        estimator : scikit-learn estimator
            The transformer or predictor to apply.

        y : dataframe, column or numpy array, optional
            The prediction targets when ``estimator`` is a supervised estimator.

        cols : string, list of strings or skrub selector, optional
            The columns to transform, when ``estimator`` is a transformer. Can
            be a column name, list of column names, or a skrub selector.

        how : "auto", "columnwise", "subframe" or "full_frame", optional
            The mode in which it is applied. In the vast majority of cases the
            default "auto" is appropriate. "columnwise" means a separate clone
            of the transformer is applied to each column. "subframe" means it
            is applied to a subset of the columns, passed as a single
            dataframe. "full_frame" means the whole input dataframe is passed
            directly to the provided ``estimator``.

        allow_reject : bool, optional
            Whether the transformer can refuse to transform columns for which
            it does not apply, in which case they are passed through unchanged.
            This can be useful to avoid specifying exactly which columns should
            be transformed. For example if we apply ``skrub.ToDatetime()`` to
            all columns with ``allow_reject=True``, string columns that can be
            parsed as dates will be converted and all other columns will be
            passed through. If we use ``allow_reject=False`` (the default), an
            error would be raised if the dataframe contains columns for which
            ``ToDatetime`` does not apply (eg a column of numbers).

        Returns
        -------
        result
            The transformed dataframe when ``estimator`` is a transformer, and
            the fitted ``estimator``'s predictions if it is a supervised
            predictor.

        Examples
        --------
        >>> import skrub

        >>> x = skrub.X(skrub.toy_orders().X)
        >>> x
        <Var 'X'>
        Result:
        ―――――――
           ID product  quantity        date
        0   1     pen         2  2020-04-03
        1   2     cup         3  2020-04-04
        2   3     cup         5  2020-04-04
        3   4   spoon         1  2020-04-05

        >>> datetime_encoder = skrub.DatetimeEncoder(add_total_seconds=False)
        >>> x.skb.apply(skrub.TableVectorizer(datetime=datetime_encoder))
        <Apply TableVectorizer>
        Result:
        ―――――――
            ID  product_cup  product_pen  ...  date_year  date_month  date_day
        0  1.0          0.0          1.0  ...     2020.0         4.0       3.0
        1  2.0          1.0          0.0  ...     2020.0         4.0       4.0
        2  3.0          1.0          0.0  ...     2020.0         4.0       4.0
        3  4.0          0.0          0.0  ...     2020.0         4.0       5.0

        Transform only the ``'product'`` column:

        >>> x.skb.apply(skrub.StringEncoder(n_components=2), cols='product') # doctest: +SKIP
        <Apply StringEncoder>
        Result:
        ―――――――
           ID     product_0     product_1  quantity        date
        0   1 -2.560113e-16  1.000000e+00         2  2020-04-03
        1   2  1.000000e+00  7.447602e-17         3  2020-04-04
        2   3  1.000000e+00  7.447602e-17         5  2020-04-04
        3   4 -3.955170e-16 -8.326673e-17         1  2020-04-05

        More complex selection of the columns to transform, here all numeric
        columns except the ``'ID'``:

        >>> from sklearn.preprocessing import StandardScaler
        >>> from skrub import selectors as s

        >>> x.skb.apply(StandardScaler(), cols=s.numeric() - "ID")
        <Apply StandardScaler>
        Result:
        ―――――――
           ID product        date  quantity
        0   1     pen  2020-04-03 -0.507093
        1   2     cup  2020-04-04  0.169031
        2   3     cup  2020-04-04  1.521278
        3   4   spoon  2020-04-05 -1.183216

        For supervised estimators, pass the targets as the argument for ``y``:

        >>> from sklearn.dummy import DummyClassifier
        >>> y = skrub.y(skrub.toy_orders().y)
        >>> y
        <Var 'y'>
        Result:
        ―――――――
        0    False
        1    False
        2     True
        3    False
        Name: delayed, dtype: bool

        >>> x.skb.apply(skrub.TableVectorizer()).skb.apply(DummyClassifier(), y=y)
        <Apply DummyClassifier>
        Result:
        ―――――――
               y
        0  False
        1  False
        2  False
        3  False
        """  # noqa: E501
        # TODO later we could also expose `wrap_transformer`'s `keep_original`
        # and `rename_cols` params
        return self._apply(
            estimator=estimator,
            y=y,
            cols=cols,
            how=how,
            allow_reject=allow_reject,
        )

    @_check_expr
    def select(self, cols):
        """Select a subset of columns.

        ``cols`` can be a column name or a list of column names, but also a
        skrub selector. Importantly, the exact list of columns that match the
        selector is stored during ``fit`` and then this same list of columns is
        selected during ``transform``.

        Parameters
        ----------
        cols : string, list of strings, or skrub selector
            The columns to select

        Returns
        -------
        dataframe with only the selected columns

        Examples
        --------
        >>> import skrub
        >>> from skrub import selectors as s
        >>> X = skrub.X(skrub.toy_orders().X)
        >>> X
        <Var 'X'>
        Result:
        ―――――――
           ID product  quantity        date
        0   1     pen         2  2020-04-03
        1   2     cup         3  2020-04-04
        2   3     cup         5  2020-04-04
        3   4   spoon         1  2020-04-05
        >>> X.skb.select(['product', 'quantity'])
        <Apply SelectCols>
        Result:
        ―――――――
          product  quantity
        0     pen         2
        1     cup         3
        2     cup         5
        3   spoon         1
        >>> X.skb.select(s.string())
        <Apply SelectCols>
        Result:
        ―――――――
          product        date
        0     pen  2020-04-03
        1     cup  2020-04-04
        2     cup  2020-04-04
        3   spoon  2020-04-05
        """
        return self._apply(SelectCols(cols), how="full_frame")

    @_check_expr
    def drop(self, cols):
        """Drop some columns.

        ``cols`` can be a column name or a list of column names, but also a
        skrub selector. Importantly, the exact list of columns that match the
        selector is stored during ``fit`` and then this same list of columns is
        dropped during ``transform``.

        Parameters
        ----------
        cols : string, list of strings, or skrub selector
            The columns to select

        Returns
        -------
        dataframe without the dropped columns

        Examples
        --------
        >>> import skrub
        >>> from skrub import selectors as s
        >>> X = skrub.X(skrub.toy_orders().X)
        >>> X
        <Var 'X'>
        Result:
        ―――――――
           ID product  quantity        date
        0   1     pen         2  2020-04-03
        1   2     cup         3  2020-04-04
        2   3     cup         5  2020-04-04
        3   4   spoon         1  2020-04-05
        >>> X.skb.drop(['ID', 'date'])
        <Apply DropCols>
        Result:
        ―――――――
          product  quantity
        0     pen         2
        1     cup         3
        2     cup         5
        3   spoon         1
        >>> X.skb.drop(s.string())
        <Apply DropCols>
        Result:
        ―――――――
           ID  quantity
        0   1         2
        1   2         3
        2   3         5
        3   4         1
        """
        return self._apply(DropCols(cols), how="full_frame")

    @_check_expr
    def concat_horizontal(self, others):
        """Concatenate dataframes horizontally.

        Parameters
        ----------
        others : list of dataframes
            The dataframes to stack horizontally with ``self``

        Returns
        -------
        dataframe
            The combined dataframes.

        Examples
        --------
        >>> import pandas as pd
        >>> import skrub
        >>> a = skrub.var('a', pd.DataFrame({'a1': [0], 'a2': [1]}))
        >>> b = skrub.var('b', pd.DataFrame({'b1': [2], 'b2': [3]}))
        >>> c = skrub.var('c', pd.DataFrame({'c1': [4], 'c2': [5]}))
        >>> a
        <Var 'a'>
        Result:
        ―――――――
           a1  a2
        0   0   1
        >>> a.skb.concat_horizontal([b, c])
        <ConcatHorizontal: 3 dataframes>
        Result:
        ―――――――
           a1  a2  b1  b2  c1  c2
        0   0   1   2   3   4   5

        Note that even if we want to concatenate a single dataframe we must
        still put it in a list:

        >>> a.skb.concat_horizontal([b])
        <ConcatHorizontal: 2 dataframes>
        Result:
        ―――――――
           a1  a2  b1  b2
        0   0   1   2   3
        """  # noqa: E501
        return Expr(ConcatHorizontal(self._expr, others))

    def clone(self, drop_values=True):
        """Get an independent clone of the expression.

        Parameters
        ----------
        drop_values : bool, default=True
            Whether to drop the initial values passed to ``skrub.var()``.
            This is convenient for example to serialize expressions without
            creating large files.

        Returns
        -------
        clone
            A new expression which does not share its state (such as fitted
            estimators) or cache with the original, and possibly without the
            variables' values.

        Examples
        --------
        >>> import skrub
        >>> c = skrub.var('a', 0) + skrub.var('b', 1)
        >>> c
        <BinOp: add>
        Result:
        ―――――――
        1
        >>> c.skb.get_data()
        {'a': 0, 'b': 1}
        >>> clone = c.skb.clone()
        >>> clone
        <BinOp: add>
        >>> clone.skb.get_data()
        {}

        We can ask to keep the variables'values:

        >>> clone = c.skb.clone(drop_values=False)
        >>> clone.skb.get_data()
        {'a': 0, 'b': 1}

        Note that in that case the cache used for previews is still cleared. So
        if we want the preview we need to prime the new expression by
        evaluating it once (either directly or by adding more steps to it):

        >>> clone
        <BinOp: add>
        >>> clone.skb.eval()
        1
        >>> clone
        <BinOp: add>
        Result:
        ―――――――
        1
        """
        from ._evaluation import clone

        return clone(self._expr, drop_preview_data=drop_values)

    def eval(self, environment=None):
        """Evaluate the expression.

        This returns the result produced by evaluating the expression, ie
        running the corresponding pipeline. The result is **always** the output
        of the pipeline's ``fit_transform`` -- the pipeline is refitted to the
        provided data.

        If no data is provided, the values passed when creating the variables
        in the expression are used.

        Parameters
        ----------
        environment : dict or None, optional
            If ``None``, the initial values of the variables contained in the
            expression are used. If a dict, it must map the name of each
            variable to a corresponding value.

        Returns
        -------
        result
            The result of running the computation, ie of executing the
            pipeline's ``fit_transform`` on the provided data.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var('a', 10)
        >>> b = skrub.var('b', 5)
        >>> c = a + b
        >>> c
        <BinOp: add>
        Result:
        ―――――――
        15
        >>> c.skb.eval()
        15
        >>> c.skb.eval({'a': 1, 'b': 2})
        3
        """
        # TODO switch position of environment and mode in _evaluation.evaluate etc.
        from ._evaluation import evaluate

        if environment is None:
            mode = "preview"
            clear = False
        else:
            mode = "fit_transform"
            clear = True
            environment = {
                **environment,
                "_skrub_use_var_values": not _expr_values_provided(
                    self._expr, environment
                ),
            }

        return evaluate(self._expr, mode=mode, environment=environment, clear=clear)

    @_check_expr
    def freeze_after_fit(self):
        """Freeze the result during pipeline fitting.

        Note this is an advanced functionality, and the need for it is usually
        an indication that we need to define a custom scikit-learn transformer
        that we can use with ``.skb.apply()``.

        When we use ``freeze_after_fit()``, the result of the expression is
        computed during ``fit()``, and then reused (not recomputed) during
        ``transform()`` or ``predict()``.

        Returns
        -------
        The expression whose value does not change after ``fit()``

        Examples
        --------
        >>> import skrub
        >>> X_df = skrub.toy_orders().X
        >>> X_df
           ID product  quantity        date
        0   1     pen         2  2020-04-03
        1   2     cup         3  2020-04-04
        2   3     cup         5  2020-04-04
        3   4   spoon         1  2020-04-05
        >>> n_products = skrub.X()['product'].nunique()
        >>> transformer = n_products.skb.get_estimator()
        >>> transformer.fit_transform({'X': X_df})
        3

        If we take only the first 2 rows ``nunique()`` (a stateless function)
        returns ``2``:

        >>> transformer.transform({'X': X_df.iloc[:2]})
        2

        If instead of recomputing it we want the number of products to be
        remembered during ``fit`` and reused during ``transform``:

        >>> n_products = skrub.X()['product'].nunique().skb.freeze_after_fit()
        >>> transformer = n_products.skb.get_estimator()
        >>> transformer.fit_transform({'X': X_df})
        3
        >>> transformer.transform({'X': X_df.iloc[:2]})
        3
        """
        return Expr(FreezeAfterFit(self._expr))

    def get_data(self):
        """Collect the values of the variables contained in the expression.

        Returns
        -------
        dict mapping variable names to their values
            Variables for which no value was given do not appear in the result.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var('a', 0)
        >>> b = skrub.var('b', 1)
        >>> c = skrub.var('c') # note no value
        >>> e = a + b + c
        >>> e.skb.get_data()
        {'a': 0, 'b': 1}
        """
        from ._evaluation import nodes

        data = {}

        for n in nodes(self._expr):
            impl = n._skrub_impl
            if isinstance(impl, Var) and impl.value is not _Constants.NO_VALUE:
                data[impl.name] = impl.value
        return data

    def draw_graph(self):
        """Get an SVG string representing the computation graph.

        In addition to the usual ``str`` methods, the result has an ``open()``
        method which displays it in a web browser window.

        Returns
        -------
        str
           SVG drawing of the computation graph.
        """
        from ._inspection import draw_expr_graph

        return draw_expr_graph(self._expr)

    def describe_steps(self):
        """Get a text representation of the computation graph.

        Usually the graphical representation provided by ``draw_graph`` or
        ``full_report`` is more useful. This is a fallback for inspecting the
        computation graph when only text output is available.

        Returns
        -------
        str
            A string representing the different computation steps, one on each
            line.

        Examples
        --------
        >>> import skrub
        >>> a = skrub.var('a')
        >>> b = skrub.var('b')
        >>> c = a + b
        >>> d = c * c
        >>> print(d.skb.describe_steps())
        VAR 'a'
        VAR 'b'
        BINOP: add
        ( VAR 'a' )*
        ( VAR 'b' )*
        ( BINOP: add )*
        BINOP: mul
        * Cached, not recomputed

        The above should be read from top to bottom as instructions for a
        simple stack machine: load the variable 'a', load the variable 'b',
        compute the addition leaving the result of (a + b) on the stack, then
        repeat this operation (but the second time no computation actually runs
        because the result of evaluating ``c`` has been cached in-memory), and
        finally evaluate the multiplication.
        """
        from ._evaluation import describe_steps

        return describe_steps(self._expr)

    def describe_param_grid(self):
        """Describe the hyper-parameters extracted from choices in the expression.

        Expressions can contain choices, ranges of possible values to be tuned
        by hyperparameter search. This function provides a description of the
        grid (set of combinations) of hyperparameters extracted from the
        expression.

        Please refer to the examples gallery for a full explanation of choices
        and hyper-parameter tuning.

        Returns
        -------
        str
            A textual description of the different choices contained in this
            expression.

        Examples
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.feature_selection import SelectKBest

        >>> import skrub

        >>> X = skrub.X()
        >>> y = skrub.y()

        >>> dim_reduction = skrub.choose_from(
        ...     {
        ...         "PCA": PCA(
        ...             n_components=skrub.choose_int(
        ...                 5, 100, log=True, name="n_components"
        ...             )
        ...         ),
        ...         "SelectKBest": SelectKBest(
        ...             k=skrub.choose_int(5, 100, log=True, name="k")
        ...         ),
        ...     },
        ...     name="dim_reduction",
        ... )
        >>> selected = X.skb.apply(dim_reduction)
        >>> classifier = skrub.choose_from(
        ...     {
        ...         "logreg": LogisticRegression(
        ...             C=skrub.choose_float(0.001, 100, log=True, name="C")
        ...         ),
        ...         "rf": RandomForestClassifier(
        ...             n_estimators=skrub.choose_int(20, 400, name="N 🌴")
        ...         ),
        ...     },
        ...     name="classifier",
        ... )
        >>> pred = selected.skb.apply(classifier, y=y)
        >>> print(pred.skb.describe_param_grid())
        - classifier: 'logreg'
          C: choose_float(0.001, 100, log=True, name='C')
          dim_reduction: 'PCA'
          n_components: choose_int(5, 100, log=True, name='n_components')
        - classifier: 'logreg'
          C: choose_float(0.001, 100, log=True, name='C')
          dim_reduction: 'SelectKBest'
          k: choose_int(5, 100, log=True, name='k')
        - classifier: 'rf'
          N 🌴: choose_int(20, 400, name='N 🌴')
          dim_reduction: 'PCA'
          n_components: choose_int(5, 100, log=True, name='n_components')
        - classifier: 'rf'
          N 🌴: choose_int(20, 400, name='N 🌴')
          dim_reduction: 'SelectKBest'
          k: choose_int(5, 100, log=True, name='k')

        Sampling a configuration for this pipeline starts by selecting an entry
        (marked by ``-``) in the list above, then a value for each of the
        hyperparameters listed (used) in that entry. For example note that the
        configurations that use the random forest do not list the
        hyperparameter ``C`` which is used only by the logistic regression.
        """
        from ._inspection import describe_param_grid

        return describe_param_grid(self._expr)

    def full_report(
        self,
        environment=None,
        open=True,
        output_dir=None,
        overwrite=False,
    ):
        """Generate a full report of the expression's evaluation.

        This creates a report showing the computation graph, and for each
        intermediate computation, some information (such as the line of code
        where it was defined) and a display of the intermediate result (or
        error).

        The pipeline is run doing a ``fit_transform``. If ``environment`` is
        provided, it is used as the bindings for the variables in the
        expression, and otherwise, the variables' ``value``s are used.

        At the moment, this creates a directory on the filesystem containing
        HTML files. The report can be displayed by visiting the contained
        ``index.html`` in a webbrowser, or passing ``open=True`` (the default)
        to this method.

        Parameters
        ----------
        environment : dict or None (default=None)
            Bindings for variables and choices contained in the expression. If
            not provided, the variables' ``value`` and the choices default
            value are used.

        open : bool (default=True)
            Whether to open the report in a webbrowser once computed.

        output_dir : str or Path or None (default=None)
            Directory where to store the report. If ``None``, a timestamped
            subdirectory will be created in the skrub data directory.

        overwrite : bool (default=False)
            What to do if the output directory already exists. If
            ``overwrite``, replace it, otherwise raise an exception.

        Returns
        -------
        dict
            The results of evaluating the expression. The keys are
            ``'result'``, ``'error'`` and ``'report_path'``. If the execution
            raised an exception, it is contained in ``'error'`` and
            ``'result'`` is ``None``. Otherwise the result produced by the
            evaluation is in ``'result'`` and ``'error'`` is ``None``. Either
            way a report is stored at the location indicated by
            ``'report_path'``.

        Examples
        --------
        >>> import skrub
        >>> c = skrub.var('a', 1) / skrub.var('b', 2)
        >>> report = c.skb.full_report(open=False)
        >>> report['result']
        0.5
        >>> report['error']
        >>> report['report_path']
        PosixPath('.../skrub_data/execution_reports/full_expr_report_.../index.html')

        We pass data:

        >>> report = c.skb.full_report({'a': 33, 'b': 11 }, open=False)
        >>> report['result']
        3.0

        And if there was an error:

        >>> report = c.skb.full_report({'a': 1, 'b': 0},open=False)
        >>> report['result']
        >>> report['error']
        ZeroDivisionError('division by zero')
        >>> report['report_path']
        PosixPath('.../skrub_data/execution_reports/full_expr_report_.../index.html')
        """  # noqa : E501
        from ._inspection import full_report

        if environment is None:
            mode = "preview"
            clear = False
        else:
            mode = "fit_transform"
            clear = True

        return full_report(
            self._expr,
            environment=environment,
            mode=mode,
            clear=clear,
            open=open,
            output_dir=output_dir,
            overwrite=overwrite,
        )

    def get_estimator(self, fitted=False):
        from ._estimator import ExprEstimator

        estimator = ExprEstimator(self.clone())
        # We need to check here even if intermediate steps have been checked,
        # because there might be in the expression some calls to functions that
        # are pickled by value by cloudpickle and that reference global
        # variables, and those global variables may have changed since the
        # expression was created.
        _check_can_be_pickled(estimator)
        if not fitted:
            return estimator
        return estimator.fit(self.get_data())

    @_check_expr
    def mark_as_x(self):
        self._expr._skrub_impl.is_X = True
        return self._expr

    @_check_expr
    def mark_as_y(self):
        self._expr._skrub_impl.is_y = True
        return self._expr

    @_check_expr
    def set_name(self, name):
        _check_name(name)
        self._expr._skrub_impl.name = name
        return self._expr

    def set_description(self, description):
        self._expr._skrub_impl.description = description
        return self._expr

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __getattr__(self, name):
        if hasattr(ApplyNamespace, name):
            attribute_error(
                self,
                name,
                (
                    f"`.skb.{name}` only exists when the last step "
                    "is a scikit-learn estimator (when the expression was "
                    "created with `.skb.apply(...)`)"
                ),
            )
        attribute_error(self, name)


class ApplyNamespace(SkrubNamespace):
    @_check_expr
    def applied_estimator(self):
        """Retrieve the estimator applied in the previous step, as an expression.

        Examples
        --------
        >>> import skrub
        >>> orders_df = skrub.toy_orders().X
        >>> features = skrub.X(orders_df).skb.apply(skrub.TableVectorizer())
        >>> fitted_vectorizer = features.skb.applied_estimator()
        >>> fitted_vectorizer
        <AppliedEstimator>
        Result:
        ―――――――
        OnSubFrame(transformer=TableVectorizer())

        Note that in order to restrict transformers to a subset of columns,
        they will be wrapped in a meta-estimator ``OnSubFrame`` or
        ``OnEachColumn`` depending if the transformer is applied to each column
        separately or not. The actual transformer can be retrieved through the
        ``transformer_`` attribute of ``OnSubFrame`` or ``transformers_``
        attribute of ``OnEachColumn`` (a dictionary mapping column names to the
        corresponding transformer).

        >>> fitted_vectorizer.transformer_
        <GetAttr 'transformer_'>
        Result:
        ―――――――
        TableVectorizer()

        >>> fitted_vectorizer.transformer_.column_to_kind_
        <GetAttr 'column_to_kind_'>
        Result:
        ―――――――
        {'ID': 'numeric', 'quantity': 'numeric', 'date': 'datetime', 'product': 'low_cardinality'}

        Here is an example of an estimator applied column-wise:

        >>> orders_df['description'] = [f'describe {p}' for p in orders_df['product']]
        >>> from skrub import selectors as s
        >>> out = skrub.X(orders_df).skb.apply(
        ...     skrub.StringEncoder(n_components=2), cols=s.string() - "date"
        ... )
        >>> fitted_vectorizer = out.skb.applied_estimator()
        >>> fitted_vectorizer
        <AppliedEstimator>
        Result:
        ―――――――
        OnEachColumn(cols=(string() - cols('date')),
                     transformer=StringEncoder(n_components=2))
        >>> fitted_vectorizer.transformers_
        <GetAttr 'transformers_'>
        Result:
        ―――――――
        {'product': StringEncoder(n_components=2), 'description': StringEncoder(n_components=2)}
        """  # noqa: E501
        return Expr(AppliedEstimator(self._expr))

    def get_grid_search(self, *, fitted=False, **kwargs):
        from sklearn.model_selection import GridSearchCV

        from ._estimator import ParamSearch
        from ._evaluation import choices

        for c in choices(self._expr).values():
            if hasattr(c, "rvs") and not isinstance(c, typing.Sequence):
                raise ValueError(
                    "Cannot use grid search with continuous numeric ranges. "
                    "Please use `get_randomized_search` or provide a number "
                    f"of steps for this range: {c}"
                )

        search = ParamSearch(self.clone(), GridSearchCV(None, None, **kwargs))
        if not fitted:
            return search
        return search.fit(self.get_data())

    def get_randomized_search(self, *, fitted=False, **kwargs):
        from sklearn.model_selection import RandomizedSearchCV

        from ._estimator import ParamSearch

        search = ParamSearch(self.clone(), RandomizedSearchCV(None, None, **kwargs))
        if not fitted:
            return search
        return search.fit(self.get_data())

    def cross_validate(self, environment=None, **kwargs):
        from ._estimator import cross_validate

        if environment is None:
            environment = self.get_data()

        return cross_validate(self.get_estimator(), environment, **kwargs)


def _check_name(name):
    if name is None:
        return
    if not isinstance(name, str):
        raise TypeError(
            f"'name' must be a string or None, got object of type: {type(name)}"
        )
    if name.startswith("_skrub_"):
        raise ValueError(
            f"names starting with '_skrub_' are reserved for skrub use, got: {name!r}."
        )


class Var(ExprImpl):
    _fields = ["name", "value"]

    def compute(self, e, mode, environment):
        if mode == "preview":
            assert not environment
            if e.value is _Constants.NO_VALUE:
                raise UninitializedVariable(
                    f"No value value has been provided for {e.name!r}"
                )
            return e.value
        if e.name in environment:
            return environment[e.name]
        if (
            environment.get("_skrub_use_var_values", False)
            and e.value is not _Constants.NO_VALUE
        ):
            return e.value
        raise UninitializedVariable(f"No value has been provided for {e.name!r}")

    def preview_if_available(self):
        return self.value

    def __repr__(self):
        return f"<Var {self.name!r}>"


def var(name, value=_Constants.NO_VALUE):
    """Create a skrub variable.

    Variables represent inputs to a machine-learning pipeline. They can be
    combined with other variables, constants, operators, function calls etc. to
    build up complex expressions, which implicitly define the pipeline.

    See the example gallery for more information about skrub pipelines.

    Parameters
    ----------
    name : str
        The name for this input. It corresponds to a key in the dictionary that
        is passed to the pipeline's ``fit()`` method (see Examples below).
        Names must be unique within a pipeline and must not start with
        ``"_skrub_"``
    value : object, optional
        Optionally, an initial value can be given to the variable. When it is
        available, it is used to provide a preview of the pipeline's results,
        to detect errors in the pipeline early, and to provide better help and
        tab-completion in interactive Python shells.

    Returns
    -------
    A skrub variable

    Examples
    --------
    Variables without a value:

    >>> import skrub
    >>> a = skrub.var('a')
    >>> a
    <Var 'a'>
    >>> b = skrub.var('b')
    >>> c = a + b
    >>> c
    <BinOp: add>
    >>> print(c.skb.describe_steps())
    VAR 'a'
    VAR 'b'
    BINOP: add

    The names of variables correspond to keys in the inputs:

    >>> c.skb.eval({'a': 10, 'b': 6})
    16

    And also to keys to the inputs to the pipeline:
    >>> estimator = c.skb.get_estimator()
    >>> estimator.fit_transform({'a': 5, 'b': 4})
    9

    When providing a value, we see what the pipeline produces for the values we
    provided:

    >>> a = skrub.var('a', 2)
    >>> b = skrub.var('b', 3)
    >>> b
    <Var 'b'>
    Result:
    ―――――――
    3
    >>> c = a + b
    >>> c
    <BinOp: add>
    Result:
    ―――――――
    5

    The values are also used as defaults for ``eval()``:

    >>> c.skb.eval()
    5

    But we can still override them. And inputs must be provided explicitly when
    using the estimator returned by `.skb.get_estimator()`.

    >>> c.skb.eval({'a': 10, 'b': 6})
    16

    Much more information about skrub variables is provided in the examples
    gallery.
    """
    if name is None:
        raise TypeError(
            "'name' for a variable cannot be None, please provide a string."
        )
    _check_name(name)
    return Expr(Var(name, value=value))


def X(value=_Constants.NO_VALUE):
    """Create a skrub variable and mark it as being ``X``.

    This is just a convenient shortcut for::

        skrub.var("X", value).skb.mark_as_x()

    Marking a variable as ``X`` tells skrub that this is the input that defines
    cross-validation splits. Please refer to the examples gallery for more
    information.

    Parameters
    ----------
    value : object
        The value passed to ``skrub.var()``, which is used for previews of the
        pipeline's outputs, cross-validation etc. as described in the
        documentation for ``skrub.var()`` and the examples gallery.

    Returns
    -------
    A skrub variable
    """
    return Expr(Var("X", value=value)).skb.mark_as_x()


def y(value=_Constants.NO_VALUE):
    return Expr(Var("y", value=value)).skb.mark_as_y()


class Value(ExprImpl):
    _fields = ["value"]

    def compute(self, e, mode, environment):
        return e.value

    def preview_if_available(self):
        return self.value

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.value.__class__.__name__}>"


@_check_expr
def as_expr(value):
    """Create an expression that evaluates to the given value.

    This wraps any object in an expression. When the expression is evaluated,
    the result is the provided value. This has a similar role as ``deferred``,
    but for any object rather than for functions.

    Parameters
    ----------
    value : object
        The result of evaluating the expression

    Returns
    -------
    An expression that evaluates to the given value

    Examples
    --------
    >>> import skrub
    >>> data_source = skrub.var('source')
    >>> data_path = skrub.as_expr(
    ...     {"local": "data.parquet", "remote": "remote/data.parquet"}
    ... )[data_source]
    >>> data_path.skb.eval({'source': 'remote'})
    'remote/data.parquet'

    Turning the dictionary into an expression defers the lookup of
    ``data_source`` until it has been evaluated when the pipeline runs.

    The example above is somewhat contrived, but ``as_expr`` is often useful
    with choices.

    >>> x1 = skrub.var('x1')
    >>> x2 = skrub.var('x2')
    >>> features = skrub.choose_from({'x1': x1, 'x2': x2}, name='features')
    >>> skrub.as_expr(features).skb.apply(skrub.TableVectorizer())
    <Apply TableVectorizer>

    In fact, this can even be shortened slightly by using the choice's method
    ``as_expr``:

    >>> features.as_expr().skb.apply(skrub.TableVectorizer())
    <Apply TableVectorizer>
    """
    return Expr(Value(value))


class IfElse(ExprImpl):
    _fields = ["condition", "value_if_true", "value_if_false"]

    def __repr__(self):
        cond = self.condition.__class__.__name__
        if_true = self.value_if_true.__class__.__name__
        if_false = self.value_if_false.__class__.__name__
        return f"<{self.__class__.__name__} {cond} ? {if_true} : {if_false}>"


@_check_expr
def if_else(condition, value_if_true, value_if_false):
    return Expr(IfElse(condition, value_if_true, value_if_false))


class FreezeAfterFit(ExprImpl):
    _fields = ["parent"]

    def fields_required_for_eval(self, mode):
        if "fit" in mode or mode == "preview":
            return self._fields
        return []

    def compute(self, e, mode, environment):
        if mode == "preview" or "fit" in mode:
            self.value_ = e.parent
        return self.value_


class _PassThrough(BaseEstimator):
    def fit(self):
        return self

    def fit_transform(self, X, y=None):
        return X

    def transform(self, X):
        return X


def _check_column_names(X):
    # NOTE: could allow int column names when how='full_frame', prob. not worth
    # the added complexity.
    #
    # TODO: maybe also forbid duplicates? use a reduced version of
    # CheckInputDataFrame? (CheckInputDataFrame does too much eg it transforms
    # numpy arrays to dataframes)
    return cast_column_names_to_strings(X)


class Apply(ExprImpl):
    _fields = ["estimator", "cols", "X", "y", "how", "allow_reject"]

    def fields_required_for_eval(self, mode):
        if "fit" in mode or mode in ["score", "preview"]:
            return self._fields
        return ["estimator", "X"]

    def compute(self, e, mode, environment):
        method_name = "fit_transform" if mode == "preview" else mode

        X = _check_column_names(e.X)

        if "fit" in method_name:
            self.estimator_ = _wrap_estimator(
                estimator=e.estimator,
                cols=e.cols,
                how=e.how,
                allow_reject=e.allow_reject,
                X=X,
            )

        if "transform" in method_name and not hasattr(self.estimator_, "transform"):
            if "fit" in method_name:
                self.estimator_.fit(X, e.y)
                if sbd.is_column(e.y):
                    self._all_outputs = [sbd.name(e.y)]
                elif sbd.is_dataframe(e.y):
                    self._all_outputs = sbd.column_names(e.y)
                else:
                    self._all_outputs = None
            pred = self.estimator_.predict(X)
            if not sbd.is_dataframe(X):
                return pred
            if len(pred.shape) == 1:
                self._all_outputs = ["y"]
                result = sbd.make_dataframe_like(X, {self._all_outputs[0]: pred})
            else:
                self._all_outputs = [f"y{i}" for i in range(pred.shape[1])]
                result = sbd.make_dataframe_like(
                    X, dict(zip(self._all_outputs, pred.T))
                )
            return sbd.copy_index(X, result)

        if "fit" in method_name or method_name == "score":
            y = (e.y,)
        else:
            y = ()
        return getattr(self.estimator_, method_name)(X, *y)

    def supports_modes(self):
        modes = ["preview", "fit_transform", "transform"]
        for name in FITTED_PREDICTOR_METHODS:
            # TODO forbid estimator being lazy?
            if hasattr(unwrap_chosen_or_default(self.estimator), name):
                modes.append(name)
        return modes

    def __repr__(self):
        estimator = unwrap_chosen_or_default(self.estimator)
        if estimator.__class__.__name__ in ["OnEachColumn", "OnSubFrame"]:
            estimator = estimator.transformer
        # estimator can be None or 'passthrough'
        if isinstance(estimator, str):
            name = repr(estimator)
        elif estimator is None:
            name = "passthrough"
        else:
            name = estimator.__class__.__name__
        return f"<{self.__class__.__name__} {name}>"


class AppliedEstimator(ExprImpl):
    "Retrieve the estimator fitted in an apply step"

    _fields = ["parent"]

    def compute(self, e, mode, environment):
        return self.parent._skrub_impl.estimator_


class GetAttr(ExprImpl):
    _fields = ["parent", "attr_name"]

    def compute(self, e, mode, environment):
        try:
            return getattr(e.parent, e.attr_name)
        except AttributeError:
            pass
        if isinstance(self.parent, Expr) and hasattr(SkrubNamespace, e.attr_name):
            comment = f"Did you mean '.skb.{e.attr_name}'?"
        else:
            comment = None
        attribute_error(e.parent, e.attr_name, comment)

    def __repr__(self):
        return f"<{self.__class__.__name__} {short_repr(self.attr_name)}>"

    def pretty_repr(self):
        return f".{_get_preview(self.attr_name)}"


class GetItem(ExprImpl):
    _fields = ["parent", "key"]

    def compute(self, e, mode, environment):
        return e.parent[e.key]

    def __repr__(self):
        return f"<{self.__class__.__name__} {short_repr(self.key)}>"

    def pretty_repr(self):
        return f"[{_get_preview(self.key)!r}]"


class Call(_CloudPickle, ExprImpl):
    _fields = [
        "func",
        "args",
        "kwargs",
        "globals",
        "closure",
        "defaults",
        "kwdefaults",
    ]
    _cloudpickle_attributes = ["func"]

    def compute(self, e, mode, environment):
        func = e.func
        if e.globals or e.closure or e.defaults:
            func = types.FunctionType(
                func.__code__,
                globals={**func.__globals__, **e.globals},
                argdefs=e.defaults,
                closure=tuple(types.CellType(c) for c in e.closure),
            )

        kwargs = (e.kwdefaults or {}) | e.kwargs
        return func(*e.args, **kwargs)

    def get_func_name(self):
        if not hasattr(self.func, "_skrub_impl"):
            name = self.func.__name__
        else:
            impl = self.func._skrub_impl
            if isinstance(impl, GetAttr):
                name = impl.attr_name
            elif isinstance(impl, GetItem):
                name = impl.key
            elif isinstance(impl, Var):
                name = impl.name
            else:
                name = type(impl).__name__
        return name

    def __repr__(self):
        name = self.get_func_name()
        return f"<{self.__class__.__name__} {name!r}>"

    def pretty_repr(self):
        return f"{_get_preview(self.func).__name__}()"


class CallMethod(ExprImpl):
    """This class allows squashing GetAttr + Call to simplify the graph."""

    _fields = ["obj", "method_name", "args", "kwargs"]

    def compute(self, e, mode, environment):
        try:
            return getattr(e.obj, e.method_name)(*e.args, **e.kwargs)
        except (TypeError, AssertionError) as err:
            # Better error message if we used the pandas DataFrame's `apply()`
            # but we meant `.skb.apply()`
            if (
                e.method_name == "apply"
                and e.args
                and isinstance(e.args[0], BaseEstimator)
            ):
                raise TypeError(
                    f"Calling `.apply()` with an estimator: `{e.args[0]!r}` "
                    "failed with the error above. Did you mean `.skb.apply()`?"
                ) from err
            else:
                raise

    def get_func_name(self):
        return self.method_name

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.method_name!r}>"

    def pretty_repr(self):
        return f".{_get_preview(self.method_name)}()"


def deferred(func):
    """Wrap function calls in an expression.

    When this decorator is applied, the resulting function returns expressions.
    The returned expression wraps the call to the original function, and the
    call is actually executed when the expression is evaluated.

    This allows including a call to any function as a step in a pipeline,
    rather than executing it immediately.

    See the examples gallery for an in-depth explanation of skrub expressions
    and ``deferred``.

    Parameters
    ----------
    func : function
        The function to wrap

    Returns
    -------
    A new function
        When called, rather than applying the original function immediately, it
        returns an expression. Evaluating the expression applies the original
        function.

    Examples
    --------
    >>> def tokenize(text):
    ...     words = text.split()
    ...     return [w for w in words if w not in ['the', 'of']]
    >>> tokenize('the first day of the week')
    ['first', 'day', 'week']

    >>> import skrub
    >>> text = skrub.var('text')

    Calling ``tokenize`` on a skrub expression raises an exception:
    ``tokenize`` tries to iterate immediately over the tokens to remove stop
    words, but the text will only be known when we run the pipeline.

    >>> tokens = tokenize(text)
    Traceback (most recent call last):
        ...
    TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly iterate over it now.

    We can defer the call to ``tokenize`` until we are evaluating the
    expression:

    >>> tokens = skrub.deferred(tokenize)(text)
    >>> tokens
    <Call 'tokenize'>
    >>> tokens.skb.eval({'text': 'the first month of the year'})
    ['first', 'month', 'year']

    Like any decorator ``deferred`` can be called explicitly as shown above or
    used with the ``@`` syntax:

    >>> @skrub.deferred
    ... def log(x):
    ...     print('INFO x =', x)
    ...     return x
    >>> x = skrub.var('x')
    >>> e = log(x)
    >>> e.skb.eval({'x': 3})
    INFO x = 3
    3

    Advanced examples
    -----------------
    As we saw in the last example above, the arguments passed to the function,
    if they are expressions, are evaluated before calling it. This is also the
    case for global variables, default arguments and free variables.

    >>> a = skrub.var('a')
    >>> b = skrub.var('b')
    >>> c = skrub.var('c')

    >>> @skrub.deferred
    ... def f(x, y=b):
    ...     z = c
    ...     print(f'{x=}, {y=}, {z=}')
    ...     return x + y + z

    >>> result = f(a)
    >>> result
    <Call 'f'>
    >>> result.skb.eval({'a': 100, 'b': 20, 'c': 3})
    x=100, y=20, z=3
    123

    Another example with a closure:

    >>> import numpy as np

    >>> def make_transformer(mode, period):
    ...
    ...     @skrub.deferred
    ...     def transform(x):
    ...         if mode == "identity":
    ...             return x[:, None]
    ...         assert mode == "trigo", mode
    ...         x = x / period * 2 * np.pi
    ...         return np.asarray([np.sin(x), np.cos(x)]).T.round(2)
    ...
    ...     return transform


    >>> hour = skrub.var("hour")
    >>> hour_encoding = skrub.choose_from(["identity", "trigo"], name="hour_encoding")
    >>> transformer = make_transformer(hour_encoding, 24)
    >>> out = transformer(hour)

    The free variable ``mode`` is evaluated before calling the deferred (inner)
    function so ``transform`` works as expected:

    >>> out.skb.eval({"hour": np.arange(0, 25, 4)})
    array([[ 0],
           [ 4],
           [ 8],
           [12],
           [16],
           [20],
           [24]])
    >>> out.skb.eval({"hour": np.arange(0, 25, 4), "hour_encoding": "trigo"})
    array([[ 0.  ,  1.  ],
           [ 0.87,  0.5 ],
           [ 0.87, -0.5 ],
           [ 0.  , -1.  ],
           [-0.87, -0.5 ],
           [-0.87,  0.5 ],
           [-0.  ,  1.  ]])
    """  # noqa : E501
    from ._evaluation import needs_eval

    @_check_call
    @_check_expr
    @functools.wraps(func)
    def deferred_func(*args, **kwargs):
        return Expr(
            Call(
                func,
                args,
                kwargs,
                globals={},
                closure=(),
                defaults=(),
                kwdefaults={},
            )
        )

    if not hasattr(func, "__code__"):
        return deferred_func

    globals_names = [
        i.argval
        for i in dis.get_instructions(func.__code__)
        if i.opname == "LOAD_GLOBAL"
    ]
    f_globals = {
        name: func.__globals__[name]
        for name in globals_names
        if name in func.__globals__
        and not isinstance(
            func.__globals__[name],
            (
                types.FunctionType,
                types.BuiltinFunctionType,
                type,
                types.ModuleType,
            ),
        )
        and needs_eval(func.__globals__[name])
    }
    closure = tuple(c.cell_contents for c in func.__closure__ or ())
    if not f_globals and not needs_eval(
        (closure, func.__defaults__, func.__kwdefaults__)
    ):
        return deferred_func

    @_check_call
    @_check_expr
    @functools.wraps(func)
    def deferred_func(*args, **kwargs):
        return Expr(
            Call(
                func,
                args,
                kwargs,
                globals=f_globals,
                closure=closure,
                defaults=func.__defaults__,
                kwdefaults=func.__kwdefaults__,
            )
        )

    return deferred_func


def deferred_optional(func, cond):
    from ._choosing import choose_bool

    if isinstance(cond, str):
        cond = choose_bool(cond)

    deferred_func = deferred(func)

    def f(*args, **kwargs):
        return cond.match(
            {True: deferred_func(*args, **kwargs), False: args[0]}
        ).as_expr()

    return f


class ConcatHorizontal(ExprImpl):
    _fields = ["first", "others"]

    def compute(self, e, mode, environment):
        if not sbd.is_dataframe(e.first):
            raise TypeError(
                "`concat_horizontal` can only be used with dataframes. "
                "`.skb.concat_horizontal` was accessed on an object of type "
                f"{e.first.__class__.__name__!r}"
            )
        if sbd.is_dataframe(e.others):
            raise TypeError(
                "`concat_horizontal` should be passed a list of dataframes."
                "If you have a single dataframe, wrap it in a list: "
                "`concat_horizontal([table_1])` not `concat_horizontal(table_1)`"
            )
        idx, non_df = next(
            ((i, o) for i, o in enumerate(e.others) if not sbd.is_dataframe(o)),
            (None, None),
        )
        if non_df is not None:
            raise TypeError(
                "`concat_horizontal` should be passed a list of dataframes: "
                "`table_0.skb.concat_horizontal([table_1, ...])`. "
                f"An object of type {non_df.__class__.__name__!r} "
                f"was found at index {idx}."
            )
        result = sbd.concat_horizontal(e.first, *e.others)
        if mode == "preview" or "fit" in mode:
            self.all_outputs_ = sbd.column_names(result)
        else:
            result = sbd.set_column_names(result, self.all_outputs_)
        return result

    def __repr__(self):
        try:
            detail = f": {len(self.others) + 1} dataframes"
        except Exception:
            detail = ""
        return f"<{self.__class__.__name__}{detail}>"


class BinOp(ExprImpl):
    _fields = ["left", "right", "op"]

    def compute(self, e, mode, environment):
        return e.op(e.left, e.right)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: {self.op.__name__.lstrip('__').rstrip('__')}>"
        )


class UnaryOp(ExprImpl):
    _fields = ["operand", "op"]

    def compute(self, e, mode, environment):
        return e.op(e.operand)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: {self.op.__name__.lstrip('__').rstrip('__')}>"
        )
