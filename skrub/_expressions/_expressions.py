import dis
import enum
import functools
import html
import inspect
import itertools
import operator
import pathlib
import reprlib
import textwrap
import traceback
import types

from sklearn.base import BaseEstimator

from .. import _dataframe as sbd
from .. import _selectors as s
from .._select_cols import DropCols, SelectCols
from .._tuning import Choice, unwrap_chosen_or_default
from .._wrap_transformer import wrap_transformer
from ._utils import FITTED_PREDICTOR_METHODS, _CloudPickle, attribute_error

__all__ = ["var", "value", "deferred", "if_else"]

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

        __init__.__signature__ = sig

        cls.__init__ = __init__
        cls._init_signature = sig

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


def _with_preview_evaluation(f):
    """Evaluate the preview of an expression after it is created.

    We decorate the functions that create it rather than do it in Expr.__init__
    to shorten the traceback when it fails, and so that we only do it when a
    user calls one of the factory functions.
    """

    @functools.wraps(f)
    def _check_preview(*args, **kwargs):
        from ._evaluation import evaluate

        expr = f(*args, **kwargs)

        try:
            expr.skb.get_data()
        except UninitializedVariable:
            pass
        try:
            evaluate(expr, mode="preview", environment=None)
        except UninitializedVariable:
            pass
        except Exception as e:
            try:
                func_name = expr._skrub_impl.pretty_repr()
            except Exception:
                func_name = f"{f.__name__}()"
            msg = "\n".join(traceback.format_exception_only(e)).rstrip("\n")
            raise RuntimeError(
                f"Preview evaluation of {func_name!r} failed.\n"
                f"You can see the full traceback above. The error message was:\n{msg}"
            ) from e

        return expr

    return _check_preview


def _get_preview(obj):
    if isinstance(obj, Expr) and "preview" in obj._skrub_impl.results:
        return obj._skrub_impl.results["preview"]
    return obj


def _with_check_call_return_value(f):
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

    @_with_preview_evaluation
    def __getattr__(self, name):
        if name in [
            "_skrub_impl",
            "get_params",
            *_EXCLUDED_STANDARD_ATTR,
            *_EXCLUDED_JUPYTER_ATTR,
        ]:
            attribute_error(self, name)
        # besides the explicitly excluded attributes, returning a GetAttr for
        # any special method is unlikely to do what we want.
        if name.startswith("__") and name.endswith("__"):
            attribute_error(self, name)
        return Expr(GetAttr(self, name))

    @_with_preview_evaluation
    def __getitem__(self, key):
        return Expr(GetItem(self, key))

    @_with_check_call_return_value
    @_with_preview_evaluation
    def __call__(self, *args, **kwargs):
        impl = self._skrub_impl
        if isinstance(impl, GetAttr):
            return Expr(CallMethod(impl.parent, impl.attr_name, args, kwargs))
        return Expr(
            Call(self, args, kwargs, globals={}, closure=(), defaults=(), kwdefaults={})
        )

    @_with_preview_evaluation
    def __len__(self):
        return Expr(GetAttr(self, "__len__"))()

    def __dir__(self):
        names = ["skb"]
        preview = self._skrub_impl.preview_if_available()
        if preview is not _Constants.NO_VALUE:
            names.extend(dir(preview))
        # TODO: unique & sort?
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
        return SkrubNamespace(self)

    def __repr__(self):
        result = repr(self._skrub_impl)
        preview = self._skrub_impl.preview_if_available()
        if preview is _Constants.NO_VALUE:
            return result
        return f"{result}\nPreview:\n――――――――\n{preview!r}"

    def _repr_html_(self):
        report = self.skb.get_report()
        if hasattr(report, "_repr_html_"):
            report = report._repr_html_()
            title = html.escape(repr(self._skrub_impl))
            # TODO: expose "summary" in TableReport?
            report = report.replace(
                '<div id="report">',
                f'<div id="report">\n<pre>{title}\nPreview:\n</pre>',
            )
        return report


def _make_bin_op(op_name):
    def op(self, right):
        return Expr(BinOp(self, right, getattr(operator, op_name)))

    op.__name__ = op_name
    return _with_preview_evaluation(op)


for op_name in _BIN_OPS:
    setattr(Expr, op_name, _make_bin_op(op_name))


def _make_r_bin_op(op_name):
    def op(self, left):
        return Expr(BinOp(left, self, getattr(operator, op_name)))

    op.__name__ = f"__r{op_name.strip('_')}__"
    return _with_preview_evaluation(op)


for op_name in _BIN_OPS:
    rop_name = f"__r{op_name.strip('_')}__"
    setattr(Expr, rop_name, _make_r_bin_op(op_name))


def _make_unary_op(op_name):
    def op(self):
        return Expr(UnaryOp(self, getattr(operator, op_name)))

    op.__name__ = op_name
    return _with_preview_evaluation(op)


for op_name in _UNARY_OPS:
    setattr(Expr, op_name, _make_unary_op(op_name))


def _wrap_estimator(estimator, cols, X):
    if estimator in [None, "passthrough"]:
        estimator = _PassThrough()
    if isinstance(estimator, Choice):
        return estimator.map_values(lambda v: _wrap_estimator(v, cols, X))
    if hasattr(estimator, "transform"):
        if not sbd.is_dataframe(X):
            if isinstance(cols, type(s.all())):
                return estimator
            # TODO better msg
            raise ValueError("Input should be a dataframe if cols is not all()")
        return wrap_transformer(estimator, cols)
    if isinstance(cols, type(s.all())):
        return estimator
    # TODO better msg
    raise ValueError("cols should only be provided if the estimator is a transformer")


class SkrubNamespace:
    def __init__(self, expr):
        self._expr = expr

    def _apply(self, estimator, y=None, cols=s.all(), name=None):
        expr = Expr(Apply(estimator, cols, self._expr, y))
        if name is not None:
            expr = expr.skb.set_name(name)
        return expr

    # TODO add a cond param to apply rather than having _tuning.optional?
    @_with_preview_evaluation
    def apply(self, estimator, y=None, cols=s.all(), name=None):
        return self._apply(estimator=estimator, y=y, cols=cols, name=name)

    @_with_preview_evaluation
    def select(self, cols, name=None):
        return self._apply(SelectCols(cols), name=name)

    @_with_preview_evaluation
    def drop(self, cols, name=None):
        return self._apply(DropCols(cols), name=name)

    @_with_preview_evaluation
    def concat_horizontal(self, others):
        return Expr(ConcatHorizontal(self._expr, others))

    def clone(self):
        from ._evaluation import clone

        return clone(self._expr)

    def get_value(self, environment=None, mode="preview"):
        # TODO switch position of environment and mode in _evaluation.evaluate etc.
        from ._evaluation import evaluate

        return evaluate(self._expr, mode=mode, environment=environment, clear=True)

    def eval(self, environment=None, mode="preview"):
        """alias for get_value -- TODO choose & keep only 1 of the 2."""
        return self.get_value(environment=environment, mode=mode)

    @property
    def preview(self):
        return self.get_value()

    @_with_preview_evaluation
    def freeze_after_fit(self):
        return Expr(FreezeAfterFit(self._expr))

    def get_data(self):
        from ._evaluation import nodes

        var_names = {}
        data = {}

        def _check_duplicate(node):
            impl = node._skrub_impl
            if (prev := var_names.get(impl.name, None)) is None:
                return
            if prev is node:
                return
            first_stack = prev._skrub_impl.creation_stack_last_line
            second_stack = impl.creation_stack_last_line
            msg = (
                f"The variable name {impl.name!r} has been used several times "
                "in this expression to refer to different quantities:\n"
                f"{impl.name!r} used here:\n"
                f"{first_stack}\n"
                f"{impl.name!r} used again here:\n"
                f"{second_stack}"
            )
            raise ValueError(msg)

        for n in nodes(self._expr):
            impl = n._skrub_impl
            if isinstance(impl, Var):
                _check_duplicate(n)
                var_names[impl.name] = n
                data[impl.name] = impl.placeholder
        return data

    def get_report(self, mode="preview", environment=None, **report_kwargs):
        from ._inspection import node_report

        return node_report(
            self._expr, mode=mode, environment=environment, **report_kwargs
        )

    def draw_graph(self):
        from ._inspection import draw_expr_graph

        return draw_expr_graph(self._expr)

    def describe_steps(self):
        from ._evaluation import describe_steps

        return describe_steps(self._expr)

    def describe_param_grid(self):
        from ._inspection import describe_param_grid

        return describe_param_grid(self._expr)

    def full_report(
        self,
        environment=None,
        mode="preview",
        open=True,
        output_dir=None,
        overwrite=False,
    ):
        from ._inspection import full_report

        return full_report(
            self._expr,
            environment=environment,
            mode=mode,
            open=open,
            output_dir=output_dir,
            overwrite=overwrite,
        )

    def _get_clone(self):
        from ._evaluation import clone

        return clone(self._expr, drop_preview_data=True)

    def get_estimator(self, fitted=False):
        from ._estimator import ExprEstimator

        estimator = ExprEstimator(self._get_clone())
        if not fitted:
            return estimator
        return estimator.fit(self.get_data())

    def get_grid_search(self, *, fitted=False, **kwargs):
        from sklearn.model_selection import GridSearchCV

        from ._estimator import ParamSearch

        search = ParamSearch(self._get_clone(), GridSearchCV(None, None, **kwargs))
        if not fitted:
            return search
        return search.fit(self.get_data())

    def get_randomized_search(self, *, fitted=False, **kwargs):
        from sklearn.model_selection import RandomizedSearchCV

        from ._estimator import ParamSearch

        search = ParamSearch(
            self._get_clone(), RandomizedSearchCV(None, None, **kwargs)
        )
        if not fitted:
            return search
        return search.fit(self.get_data())

    def cross_validate(self, environment=None, **kwargs):
        from ._estimator import cross_validate

        if environment is None:
            environment = self.get_data()

        return cross_validate(self.get_estimator(), environment, **kwargs)

    def mark_as_x(self):
        # TODO copy instead?
        # TODO check that at most 1 node is marked as X and y
        self._expr._skrub_impl.is_X = True
        return self._expr

    def mark_as_y(self):
        self._expr._skrub_impl.is_y = True
        return self._expr

    def set_name(self, name):
        _check_name(name)
        self._expr._skrub_impl.name = name
        return self._expr

    def describe(self, description):
        self._expr._skrub_impl.description = description
        return self._expr

    def clear_results(self):
        from ._evaluation import clear_results

        clear_results(self._expr)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


def _check_name(name):
    if name is None:
        return
    if not isinstance(name, str):
        raise TypeError(f"'name' must be a string or None, got: {name!r}")
    if name.startswith("_skrub_"):
        raise ValueError(
            f"names starting with '_skrub_' are reserved for skrub use, got: {name!r}."
        )


class Var(ExprImpl):
    _fields = ["name", "placeholder"]

    def compute(self, e, mode, environment):
        if mode == "preview":
            # still use value from the environment if provided, fallback on
            # placeholder otherwise.
            if e.name in environment:
                return environment[e.name]
            if e.placeholder is _Constants.NO_VALUE:
                raise UninitializedVariable(
                    f"No placeholder value has been provided for {e.name!r}"
                )
            return e.placeholder
        if e.name not in environment:
            raise UninitializedVariable(f"No value has been provided for {e.name!r}")
        return environment[e.name]

    def preview_if_available(self):
        return self.placeholder

    def __repr__(self):
        return f"<Var {self.name!r}>"


def var(name, placeholder=_Constants.NO_VALUE):
    if name is None:
        raise TypeError(
            "'name' for a variable cannot be None, please provide a string."
        )
    _check_name(name)
    return Expr(Var(name, placeholder=placeholder))


def X(placeholder=_Constants.NO_VALUE):
    return Expr(Var("X", placeholder=placeholder)).skb.mark_as_x()


def y(placeholder=_Constants.NO_VALUE):
    return Expr(Var("y", placeholder=placeholder)).skb.mark_as_y()


class Value(ExprImpl):
    _fields = ["value"]

    def compute(self, e, mode, environment):
        return e.value

    def preview_if_available(self):
        return self.value

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.value.__class__.__name__}>"


def value(value):
    return Expr(Value(value))


class IfElse(ExprImpl):
    _fields = ["condition", "value_if_true", "value_if_false"]

    def __repr__(self):
        cond = self.condition.__class__.__name__
        if_true = self.value_if_true.__class__.__name__
        if_false = self.value_if_false.__class__.__name__
        return f"<{self.__class__.__name__} {cond} ? {if_true} : {if_false}>"


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
            # TODO: have separate fitted attributes for preview?
            # (same question for Apply)
            self.value_ = e.parent
        return self.value_


class _PassThrough(BaseEstimator):
    def fit(self):
        return self

    def fit_transform(self, X, y=None):
        return X

    def transform(self, X):
        return X


class Apply(ExprImpl):
    _fields = ["estimator", "cols", "X", "y"]

    def fields_required_for_eval(self, mode):
        if "fit" in mode or mode in ["score", "preview"]:
            return self._fields
        return ["estimator", "cols", "X"]

    def compute(self, e, mode, environment):
        method_name = "fit_transform" if mode == "preview" else mode

        if "fit" in method_name:
            self.estimator_ = _wrap_estimator(e.estimator, e.cols, e.X)

        if "transform" in method_name and not hasattr(self.estimator_, "transform"):
            if "fit" in method_name:
                self.estimator_.fit(e.X, e.y)
                if sbd.is_column(e.y):
                    self._all_outputs = [sbd.name(e.y)]
                elif sbd.is_dataframe(e.y):
                    self._all_outputs = sbd.column_names(e.y)
                else:
                    self._all_outputs = None
            pred = self.estimator_.predict(e.X)
            if not sbd.is_dataframe(e.X):
                return pred
            if len(pred.shape) == 1:
                self._all_outputs = ["y"]
                result = sbd.make_dataframe_like(e.X, {self._all_outputs[0]: pred})
            else:
                self._all_outputs = [f"y{i}" for i in range(pred.shape[1])]
                result = sbd.make_dataframe_like(
                    e.X, dict(zip(self._all_outputs, pred.T))
                )
            return sbd.copy_index(e.X, result)

        if "fit" in method_name or method_name == "score":
            y = (e.y,)
        else:
            y = ()
        return getattr(self.estimator_, method_name)(e.X, *y)

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
        return f"<{self.__class__.__name__} {estimator.__class__.__name__}>"


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
        return f"<{self.__class__.__name__} {self.attr_name!r}>"

    def pretty_repr(self):
        return f".{_get_preview(self.attr_name)}"


def _get_repr_formatter():
    r = reprlib.Repr()
    r.maxlevel = 2
    r.maxtuple = 2
    r.maxlist = 2
    r.maxarray = 3
    r.maxdict = 2
    r.maxset = 2
    r.maxfrozenset = 2
    r.maxdeque = 2
    r.maxstring = 30
    r.maxlong = 10
    r.maxother = 30
    return r


class GetItem(ExprImpl):
    _fields = ["parent", "key"]

    def compute(self, e, mode, environment):
        return e.parent[e.key]

    def __repr__(self):
        if isinstance(self.key, Expr):
            return f"<{self.__class__.__name__} ...>"
        r = _get_repr_formatter()
        return f"<{self.__class__.__name__} {r.repr(self.key)}>"

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
        return getattr(e.obj, e.method_name)(*e.args, **e.kwargs)

    def get_func_name(self):
        return self.method_name

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.method_name!r}>"

    def pretty_repr(self):
        return f".{_get_preview(self.method_name)}()"


def deferred(func):
    if not hasattr(func, "__code__"):

        @_with_check_call_return_value
        @_with_preview_evaluation
        @functools.wraps(func)
        def inner(*args, **kwargs):
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

        return inner

    @_with_check_call_return_value
    @_with_preview_evaluation
    @functools.wraps(func)
    def inner(*args, **kwargs):
        globals_names = [
            i.argval
            for i in dis.get_instructions(func.__code__)
            if i.opname == "LOAD_GLOBAL"
        ]
        f_globals = {
            name: func.__globals__[name]
            for name in globals_names
            if name in func.__globals__
            # TODO rather than exclude modules, types & functions, check if contains
            # an expression and include only then?
            and not isinstance(
                func.__globals__[name],
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    type,
                    types.ModuleType,
                ),
            )
        }
        closure = tuple(c.cell_contents for c in func.__closure__ or ())
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

    return inner


def deferred_optional(func, cond):
    from .._tuning import choose_bool

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
