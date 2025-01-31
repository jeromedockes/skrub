from functools import partial, wraps

import sklearn
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, clone

from .. import _join_utils
from .._parallel_plot import DEFAULT_COLORSCALE, plot_parallel_coord
from .._tuning import Choice, unwrap, unwrap_default
from ._evaluation import (
    choices,
    evaluate,
    find_node_by_name,
    find_X,
    find_y,
    get_params,
    nodes,
    param_grid,
    reachable,
    set_params,
)
from ._evaluation import clone as clone_expr
from ._expressions import Apply
from ._utils import FITTED_PREDICTOR_METHODS, X_NAME, Y_NAME, attribute_error


def _prune_cache(expr, mode, *args, **kwargs):
    reachable_nodes = reachable(expr, mode)
    for node in nodes(expr):
        if id(node) not in reachable_nodes:
            node._skrub_impl.results.pop(mode, None)


def _check_env(environment, caller_name):
    """Helper to detect the mistake eg fit(X) instead of fit({'X': X})"""
    if not isinstance(environment, dict):
        raise TypeError(
            f"The first argument to {caller_name!r} should be a dictionary of input"
            f" values, for example: {caller_name}({{'X': df, 'other_table_name':"
            " other_df, ...})"
        )


class ExprEstimator(BaseEstimator):
    def __init__(self, expr):
        self.expr = expr

    def _sklearn_compatible_estimator(self):
        return CompatibleExprEstimator(clone_expr(self.expr))

    def fit(self, environment):
        _check_env(environment, "fit")
        _ = self.fit_transform(environment)
        return self

    def fit_transform(self, environment):
        _check_env(environment, "fit_transform")
        callback = partial(_prune_cache, self.expr, "fit_transform")
        env = environment | {"_callback": callback}
        return evaluate(self.expr, "fit_transform", env, clear=True)

    def _eval_in_mode(self, mode, environment):
        _check_env(environment, mode)
        callback = partial(_prune_cache, self.expr, mode)
        env = environment | {"_callback": callback}
        return evaluate(self.expr, mode, env, clear=True)

    def __getattr__(self, name):
        if name not in self.expr._skrub_impl.supports_modes():
            attribute_error(self, name)

        def f(*args, **kwargs):
            return self._eval_in_mode(name, *args, **kwargs)

        f.__name__ = name
        return f

    def get_params(self, deep=True):
        params = {"expr": self.expr}
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

    def sub_estimator(self, name):
        node = find_node_by_name(self.expr, name)
        if node is None:
            return None
        impl = node._skrub_impl
        if not isinstance(impl, Apply):
            raise TypeError(
                f"node {name!r} does not represent a sub-estimator: {node!r}"
            )
        if not hasattr(impl, "estimator_"):
            raise ValueError(
                f"Node {name!r} has not been fitted. Call fit() on the estimator "
                "before attempting to retrieve fitted sub-estimators."
            )
        return node._skrub_impl.estimator_


class CompatibleExprEstimator(ExprEstimator):
    @property
    def _estimator_type(self):
        try:
            return unwrap_default(self.expr._skrub_impl.estimator)._estimator_type
        except AttributeError:
            return "transformer"

    @property
    def classes_(self):
        try:
            estimator = self.expr._skrub_impl.estimator_
        except AttributeError:
            attribute_error(self, "classes_")
        return estimator.classes_

    def fit_transform(self, X, y=None, environment=None):
        callback = partial(_prune_cache, self.expr, "fit_transform")
        xy_environment = {X_NAME: X, "_callback": callback}
        if y is not None:
            xy_environment[Y_NAME] = y
        xy_environment = {**(environment or {}), **xy_environment}
        return evaluate(self.expr, "fit_transform", xy_environment, clear=True)

    def fit(self, X, y=None, environment=None):
        _ = self.fit_transform(X, y=y, environment=environment)
        return self

    def _eval_in_mode(self, mode, X, y=None, *, environment):
        callback = partial(_prune_cache, self.expr, mode)
        xy_environment = {X_NAME: X, "_callback": callback}
        if y is not None:
            xy_environment[Y_NAME] = y
        xy_environment = {**(environment or {}), **xy_environment}
        return evaluate(self.expr, mode, xy_environment, clear=True)


class ScorerWrapper:
    def __init__(self, scorer, environment):
        self._scorer = scorer
        self._environment = environment

    def __call__(self, estimator, X, y, *args, **kwargs):
        all_method_names = FITTED_PREDICTOR_METHODS
        for method_name in all_method_names:
            try:
                original_method = getattr(estimator, method_name)
            except AttributeError:
                continue

            @wraps(original_method)
            def wrapped(X, *y, original_method=original_method):
                return original_method(X, *y, environment=self._environment)

            setattr(estimator, method_name, wrapped)
        try:
            return self._scorer(estimator, X, y, *args, **kwargs)
        finally:
            for method_name in all_method_names:
                # TODO: maybe before patching check if some of the methods are
                # in the estimator's __dict__ and restore them after in the off
                # chance that eg predict is a callable in the dict rather than
                # a method.
                try:
                    delattr(estimator, method_name)
                except AttributeError:
                    pass


def _with_metadata_routing(func):
    @wraps(func)
    def in_context(*args, **kwargs):
        with sklearn.config_context(enable_metadata_routing=True):
            return func(*args, **kwargs)

    return in_context


@_with_metadata_routing
def cross_validate(expr_estimator, environment, scoring=None, **cv_params):
    expr = expr_estimator.expr
    X_y = _find_X_y(expr)
    X = evaluate(clone_expr(X_y["X"]), "fit_transform", environment)
    if "y" in X_y:
        y = evaluate(clone_expr(X_y["y"]), "fit_transform", environment)
    else:
        y = None

    estimator = _to_compatible(expr_estimator)

    scorer = metrics.check_scoring(estimator, scoring)
    scorer = ScorerWrapper(scorer, environment)
    estimator.set_fit_request(environment=True)
    return model_selection.cross_validate(
        estimator,
        X,
        y,
        params={"environment": environment},
        scoring=scorer,
        **cv_params,
    )


def _find_X_y(expr):
    x_node = find_X(expr)
    if x_node is None:
        raise ValueError('expr should have a node marked with "mark_as_x()"')
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


def _to_compatible(estimator):
    try:
        return estimator._sklearn_compatible_estimator()
    except AttributeError:
        return clone(estimator)


# TODO with ParameterGrid and ParameterSampler we can generate the list of
# candidates so we can provide more than just a score, eg full predictions for
# each sampled param combination.


class ParamSearch(BaseEstimator):
    def __init__(self, expr, search):
        self.expr = expr
        self.search = search

    def _sklearn_compatible_estimator(self):
        return CompatibleParamSearch(clone_expr(self.expr), clone(self.search))

    @_with_metadata_routing
    def fit(self, environment):
        X_y = _find_X_y(self.expr)
        X = evaluate(clone_expr(X_y["X"]), "fit_transform", environment)
        if "y" in X_y:
            y = evaluate(clone_expr(X_y["y"]), "fit_transform", environment)
        else:
            y = None
        self.estimator_ = CompatibleExprEstimator(clone_expr(self.expr))
        self.search_ = clone(self.search)
        self.estimator_.set_fit_request(environment=True)
        self.search_.estimator = self.estimator_
        param_grid = self._get_param_grid()
        if hasattr(self.search_, "param_grid"):
            self.search_.param_grid = param_grid
        else:
            assert hasattr(self.search_, "param_distributions")
            self.search_.param_distributions = param_grid
        scorer = metrics.check_scoring(self.estimator_, self.search.scoring)
        scorer = ScorerWrapper(scorer, environment)
        self.search_.scoring = scorer
        try:
            self.search_.fit(X, y, environment=environment)
        finally:
            # TODO copy useful attributes and stop storing self.search_ instead
            self.search_.scoring.environment = None
        return self

    def _get_param_grid(self):
        grid = param_grid(self.estimator_.expr)
        new_grid = []
        for subgrid in grid:
            subgrid = {f"expr__{k}": v for k, v in subgrid.items()}
            new_grid.append(subgrid)
        return new_grid

    def __getattr__(self, name):
        if name == "search_":
            attribute_error(self, name)
        if name not in self.expr._skrub_impl.supports_modes():
            return getattr(self.search_, name)

        def f(*args, **kwargs):
            return self._call_predictor_method(name, *args, **kwargs)

        f.__name__ = name
        return f

    def _call_predictor_method(self, name, environment):
        if not hasattr(self, "search_"):
            raise ValueError("Search not fitted")
        return getattr(self.best_estimator_, name)(environment)

    @property
    def best_estimator_(self):
        if not hasattr(self, "search_"):
            attribute_error(self, "best_estimator_")
        return ExprEstimator(self.search_.best_estimator_.expr)

    def get_cv_results_table(self, return_metadata=False, detailed=False):
        import pandas as pd

        expr_choices = choices(self.estimator_.expr)

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
        table = table.sort_values("mean_test_score", ascending=False, ignore_index=True)
        return (table, metadata) if return_metadata else table

    def plot_parallel_coord(self, colorscale=DEFAULT_COLORSCALE, min_score=None):
        cv_results, metadata = self.get_cv_results_table(
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


class CompatibleParamSearch(ParamSearch):
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

    def fit(self, X, y=None, environment=None):
        xy_environment = {X_NAME: X}
        if y is not None:
            xy_environment[Y_NAME] = y
        xy_environment = {**(environment or {}), **xy_environment}
        super().fit(xy_environment)
        return self

    def _call_predictor_method(self, name, X, y=None, *, environment):
        if not hasattr(self, "search_"):
            raise ValueError("Search not fitted")
        return getattr(self.search_.best_estimator_, name)(
            X, y=y, environment=environment
        )
