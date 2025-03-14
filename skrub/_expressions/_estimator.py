import pandas as pd
from sklearn import model_selection
from sklearn.base import BaseEstimator, clone

from .. import _join_utils
from ._choosing import Choice, unwrap, unwrap_default
from ._evaluation import (
    choices,
    evaluate,
    find_node_by_name,
    find_X,
    find_y,
    get_params,
    param_grid,
    set_params,
)
from ._expressions import Apply
from ._parallel_coord import DEFAULT_COLORSCALE, plot_parallel_coord
from ._utils import X_NAME, Y_NAME, attribute_error


class _SharedDict(dict):
    def __deepcopy__(self, memo):
        return self

    def __sklearn_clone__(self):
        return self


class ExprEstimator(BaseEstimator):
    def __init__(self, expr):
        self.expr = expr

    def __skrub_to_Xy_estimator__(self, environment):
        return XyExprEstimator(self.expr, _SharedDict(environment))

    def fit(self, environment):
        _ = self.fit_transform(environment)
        return self

    def fit_transform(self, environment):
        # TODO: not needed, can be handled by _eval_in_mode?
        return evaluate(self.expr, "fit_transform", environment, clear=True)

    def _eval_in_mode(self, mode, environment):
        return evaluate(self.expr, mode, environment, clear=True)

    def report(self, mode, environment, **full_report_kwargs):
        from ._inspection import full_report

        full_report_kwargs["clear"] = True
        return full_report(
            self.expr, environment=environment, mode=mode, **full_report_kwargs
        )

    def __getattr__(self, name):
        if name not in self.expr._skrub_impl.supports_modes():
            attribute_error(self, name)

        def f(*args, **kwargs):
            return self._eval_in_mode(name, *args, **kwargs)

        f.__name__ = name
        return f

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if not deep:
            return params
        params.update({f"expr__{k}": v for k, v in get_params(self.expr).items()})
        return params

    def set_params(self, **params):
        params = {k: unwrap(v) for k, v in params.items()}
        if "expr" in params:
            self.expr = params.pop("expr")
        set_params(self.expr, {int(k.lstrip("expr__")): v for k, v in params.items()})
        return self

    def find_fitted_estimator(self, name):
        node = find_node_by_name(self.expr, name)
        if node is None:
            return None
        impl = node._skrub_impl
        if not isinstance(impl, Apply):
            raise TypeError(
                f"Node {name!r} does not represent "
                f"the application of an estimator: {node!r}"
            )
        if not hasattr(impl, "estimator_"):
            raise ValueError(
                f"Node {name!r} has not been fitted. Call fit() on the estimator "
                "before attempting to retrieve fitted sub-estimators."
            )
        return node._skrub_impl.estimator_


def _to_Xy_estimator(estimator, environment):
    return estimator.__skrub_to_Xy_estimator__(environment)


def _to_env_estimator(estimator):
    return estimator.__skrub_to_env_estimator__()


class _XyEstimatorMixin:
    def _get_env(self, X, y):
        xy_environment = {X_NAME: X}
        if y is not None:
            xy_environment[Y_NAME] = y
        return {**self.environment, **xy_environment}


class XyExprEstimator(_XyEstimatorMixin, ExprEstimator):
    def __init__(self, expr, environment):
        self.expr = expr
        self.environment = environment

    def __skrub_to_env_estimator__(self):
        return ExprEstimator(self.expr)

    @property
    def _estimator_type(self):
        try:
            return unwrap_default(self.expr._skrub_impl.estimator)._estimator_type
        except AttributeError:
            return "transformer"

    @property
    def __sklearn_tags__(self):
        try:
            return unwrap_default(self.expr._skrub_impl.estimator).__sklearn_tags__
        except AttributeError:
            attribute_error(self, "__sklearn_tags__")

    @property
    def classes_(self):
        try:
            estimator = self.expr._skrub_impl.estimator_
        except AttributeError:
            attribute_error(self, "classes_")
        return estimator.classes_

    def fit_transform(self, X, y=None):
        return evaluate(self.expr, "fit_transform", self._get_env(X, y), clear=True)

    def fit(self, X, y=None):
        _ = self.fit_transform(X, y=y)
        return self

    def _eval_in_mode(self, mode, X, y=None):
        return evaluate(self.expr, mode, self._get_env(X, y), clear=True)


def cross_validate(expr_estimator, environment, **cv_params):
    """Cross-validate an estimator built from an expression.

    This runs cross-validation from an estimator that was built from a skrub
    expression with ``.skb.get_estimator()``, ``.skb.get_grid_search()`` or
    ``.skb.get_randomized_search()``.

    It is useful to run nested cross-validation of a grid search or randomized
    search.

    Parameters
    ----------
    expr_estimator : estimator
        An estimator generated from a skrub expression.

    environment : dict or None
        Bindings for variables contained in the expression. If not
        provided, the ``value``s passed when initializing ``var()`` are
        used.

    cv_params : dict
        All other named arguments are forwarded to
        ``sklearn.model_selection.cross_validate``.

    Returns
    -------
    dict
        Cross-validation results.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> import skrub

    >>> X_a, y_a = make_classification(random_state=0)
    >>> X, y = skrub.X(X_a), skrub.y(y_a)
    >>> log_reg = LogisticRegression(
    ...     **skrub.choose_float(0.01, 1.0, log=True, name="C")
    ... )
    >>> pred = X.skb.apply(log_reg, y=y)
    >>> search = pred.skb.get_randomized_search(random_state=0)
    >>> skrub.cross_validate(search, pred.skb.get_data())['test_score'] # doctest: +SKIP
    array([0.75, 0.95, 0.85, 0.85, 0.85])
    """
    estimator = _to_Xy_estimator(expr_estimator, environment)
    X, y = _compute_Xy(expr_estimator.expr, environment)
    result = model_selection.cross_validate(
        estimator,
        X,
        y,
        **cv_params,
    )
    if (estimators := result.get("estimator", None)) is None:
        return result
    result["estimator"] = [_to_env_estimator(e) for e in estimators]
    return result


def _find_Xy(expr):
    x_node = find_X(expr)
    if x_node is None:
        raise ValueError('expr should have a node marked with "mark_as_X()"')
    result = {"X": x_node}
    if (y_node := find_y(expr)) is not None:
        result["y"] = y_node
    else:
        impl = expr._skrub_impl
        if getattr(impl, "y", None) is not None:
            # the final estimator requests a y so some node must have been
            # marked as y
            raise ValueError('expr should have a node marked with "mark_as_y()"')
    return result


def _compute_Xy(expr, environment):
    Xy = _find_Xy(expr.skb.clone())
    X = evaluate(
        Xy["X"],
        mode="fit_transform",
        environment=environment,
        clear=False,
    )
    if "y" in Xy:
        y = evaluate(
            Xy["y"],
            mode="fit_transform",
            environment=environment,
            clear=False,
        )
    else:
        y = None
    return X, y


# TODO with ParameterGrid and ParameterSampler we can generate the list of
# candidates so we can provide more than just a score, eg full predictions for
# each sampled param combination.


class ParamSearch(BaseEstimator):
    def __init__(self, expr, search):
        self.expr = expr
        self.search = search

    def __skrub_to_Xy_estimator__(self, environment):
        return XyParamSearch(self.expr, self.search, _SharedDict(environment))

    def fit(self, environment):
        estimator = XyExprEstimator(self.expr, _SharedDict(environment))
        search = clone(self.search)
        search.estimator = estimator
        param_grid = self._get_param_grid()
        if hasattr(search, "param_grid"):
            search.param_grid = param_grid
        else:
            assert hasattr(search, "param_distributions")
            search.param_distributions = param_grid
        X, y = _compute_Xy(self.expr, environment)
        search.fit(X, y)
        self.best_estimator_ = _to_env_estimator(search.best_estimator_)
        self.cv_results_ = search.cv_results_
        return self

    def _get_param_grid(self):
        grid = param_grid(self.expr)
        new_grid = []
        for subgrid in grid:
            subgrid = {f"expr__{k}": v for k, v in subgrid.items()}
            new_grid.append(subgrid)
        return new_grid

    def __getattr__(self, name):
        if name not in self.expr._skrub_impl.supports_modes():
            attribute_error(self, name)

        def f(*args, **kwargs):
            return self._call_predictor_method(name, *args, **kwargs)

        f.__name__ = name
        return f

    def _call_predictor_method(self, name, environment):
        if not hasattr(self, "search_"):
            raise ValueError("Search not fitted")
        return getattr(self.best_estimator_, name)(environment)

    @property
    def results_(self):
        return self._get_cv_results_table()

    def _get_cv_results_table(self, return_metadata=False, detailed=False):
        expr_choices = choices(self.best_estimator_.expr)

        all_rows = []
        param_names = set()
        log_scale_columns = set()
        for params in self.cv_results_["params"]:
            row = {}
            for param_id, param in params.items():
                choice = expr_choices[int(param_id.lstrip("expr__"))]
                if isinstance(choice, Choice):
                    param = choice.outcomes[param]
                choice_name = param.in_choice or param_id
                value = param.name or param.value
                row[choice_name] = value
                param_names.add(choice_name)
                if getattr(param, "is_from_log_scale", False):
                    log_scale_columns.add(choice_name)
            all_rows.append(row)

        metadata = {"log_scale_columns": list(log_scale_columns)}
        # all_ordered_param_names = _get_all_param_names(self._get_param_grid())
        # ordered_param_names = [n for n in all_ordered_param_names if n in param_names]
        # table = pd.DataFrame(all_rows, columns=ordered_param_names)
        table = pd.DataFrame(all_rows)
        result_keys = [
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
            "mean_train_score",
            "std_train_score",
        ]
        new_names = _join_utils.pick_column_names(table.columns, result_keys)
        renaming = dict(zip(table.columns, new_names))
        table.columns = new_names
        metadata["log_scale_columns"] = [
            renaming[c] for c in metadata["log_scale_columns"]
        ]
        table.insert(0, "mean_test_score", self.cv_results_["mean_test_score"])
        if detailed:
            for k in result_keys[1:]:
                if k in self.cv_results_:
                    table.insert(table.shape[1], k, self.cv_results_[k])
        table = table.sort_values(
            "mean_test_score", ascending=False, ignore_index=True, kind="stable"
        )
        return (table, metadata) if return_metadata else table

    def plot_results(self, *, colorscale=DEFAULT_COLORSCALE, min_score=None):
        cv_results, metadata = self._get_cv_results_table(
            return_metadata=True, detailed=True
        )
        cv_results = cv_results.drop(
            [
                "std_test_score",
                "std_fit_time",
                "std_score_time",
                "mean_train_score",
                "std_train_score",
            ],
            axis="columns",
            errors="ignore",
        )
        if min_score is not None:
            cv_results = cv_results[cv_results["mean_test_score"] >= min_score]
        return plot_parallel_coord(cv_results, metadata, colorscale=colorscale)


def _get_all_param_names(grid):
    names = {}
    for subgrid in grid:
        for k, v in subgrid.items():
            if v.name is not None:
                k = v.name
            names[k] = None
    return list(names)


class XyParamSearch(_XyEstimatorMixin, ParamSearch):
    def __init__(self, expr, search, environment):
        self.expr = expr
        self.search = search
        self.environment = environment

    def __skrub_to_env_estimator__(self):
        env_estimator = ParamSearch(self.expr, self.search)
        if hasattr(self, "best_estimator_"):
            env_estimator.best_estimator_ = self.best_estimator_
            env_estimator.cv_results_ = self.cv_results_
        return env_estimator

    @property
    def _estimator_type(self):
        try:
            return unwrap_default(self.expr._skrub_impl.estimator)._estimator_type
        except AttributeError:
            return "transformer"

    @property
    def classes_(self):
        try:
            estimator = self.best_estimator_.expr._skrub_impl.estimator_
        except AttributeError:
            attribute_error(self, "classes_")
        return estimator.classes_

    def fit(self, X, y=None):
        super().fit(self._get_env(X, y))
        return self

    def _call_predictor_method(self, name, X, y=None):
        if not hasattr(self, "best_estimator_"):
            raise ValueError("Search not fitted")
        return getattr(self.best_estimator_, name)(self._get_env(X, y))
