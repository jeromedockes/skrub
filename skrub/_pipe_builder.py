import dataclasses
import io
import itertools
import re
import traceback
from typing import Any, Sequence

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from . import _dataframe as sbd
from . import _join_utils
from . import selectors as s
from ._parallel_plot import DEFAULT_COLORSCALE, plot_parallel_coord
from ._reporting import TableReport
from ._select_cols import Drop
from ._tuning import (
    Choice,
    Optional,
    choose_float,
    choose_from,
    choose_int,
    contains_choice,
    expand_grid,
    grid_description,
    optional,
    params_description,
    set_params_to_default,
    unwrap,
    unwrap_default,
    write_indented,
)
from ._wrap_transformer import wrap_transformer

__all__ = [
    "PipeBuilder",
    "Chain",
    "Recipe",
    "choose_from",
    "optional",
    "choose_float",
    "choose_int",
]


def _camel_to_snake(name):
    name = re.sub(
        r"(.)([A-Z][a-z0-9]+)",
        r"\1_\2",
        name,
    )
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def _squeeze(df):
    if sbd.shape(df)[1] == 1:
        return sbd.col(df, sbd.column_names(df)[0])
    return df


@dataclasses.dataclass
class Step:
    estimator: Any
    cols: s.Selector
    name: str | None = None
    keep_original: bool = False
    rename_columns: str = "{}"
    allow_reject: bool = False

    def _make_transformer(self, estimator=None, n_jobs=1):
        if estimator is None:
            estimator = self.estimator
        return wrap_transformer(
            estimator,
            self.cols,
            keep_original=self.keep_original,
            rename_columns=self.rename_columns,
            allow_reject=self.allow_reject,
            n_jobs=n_jobs,
        )


class NamedParamPipeline(Pipeline):
    def set_params(self, **params):
        params = {k: unwrap(v) for k, v in params.items()}
        super().set_params(**params)
        return self


def _pick_names(suggested_names):
    used = set()
    new_names = []
    for name in suggested_names:
        suffixes = map("_{}".format, itertools.count(1))
        new = name
        while new in used:
            new = f"{name}{next(suffixes)}"
        used.add(new)
        new_names.append(new)
    return new_names


def _get_name(step):
    return (
        getattr(step, "name", None)
        or getattr(step.estimator, "name", None)
        or _camel_to_snake(unwrap_default(step.estimator).__class__.__name__)
    )


def _is_passthrough(estimator):
    return (
        estimator is None or isinstance(estimator, str) and estimator == "passthrough"
    )


def _check_passthrough(estimator):
    if _is_passthrough(estimator):
        return "passthrough"
    return estimator


def _check_estimator(estimator, step, n_jobs):
    if hasattr(estimator, "predict"):
        return estimator
    if _is_passthrough(estimator):
        return "passthrough"
    return step._make_transformer(estimator, n_jobs=n_jobs)


def _to_estimator(step, n_jobs):
    estimator = step.estimator
    if isinstance(estimator, Choice):
        return estimator.map_values(lambda v: _check_estimator(v, step, n_jobs))
    return _check_estimator(estimator, step, n_jobs)


class PipeBuilder:
    def __init__(
        self,
        input_data=None,
        y_cols=(),
        n_jobs=None,
        memory=None,
        preview_sample_size=500,
        random_seed=0,
        shuffle_split=True,
    ):
        self.input_data = input_data
        self.y_cols = y_cols
        if input_data is None:
            self.input_data_shape = None
        else:
            self.input_data_shape = sbd.shape(input_data)
        self.n_jobs = n_jobs
        self.memory = memory
        self.preview_sample_size = preview_sample_size
        self.random_seed = random_seed
        self.shuffle_split = shuffle_split
        self._steps = [
            Step(
                estimator=Drop(),
                cols=(s.all() & self.y_cols),
                name="_drop_y_columns",
            )
        ]

    def _with_prepared_steps(self, steps):
        new = self.__class__(
            input_data=self.input_data,
            y_cols=self.y_cols,
            n_jobs=self.n_jobs,
            memory=self.memory,
            preview_sample_size=self.preview_sample_size,
            random_seed=self.random_seed,
        )
        new._steps = steps
        return new

    def _has_predictor(self):
        if not self._steps:
            return False
        return hasattr(unwrap_default(self._steps[-1].estimator), "predict")

    def _get_step_names(self):
        suggested_names = [_get_name(step) for step in self._steps]
        return _pick_names(suggested_names)

    def _get_estimators(self):
        return [_to_estimator(step, self.n_jobs) for step in self._steps]

    def _get_default_estimators(self):
        return [
            set_params_to_default(unwrap_default(estimator))
            for estimator in self._get_estimators()
        ]

    def get_param_grid(self):
        grid = {
            name: estimator
            for (name, estimator) in zip(self._get_step_names(), self._get_estimators())
            if contains_choice(estimator)
        }
        return expand_grid(grid)

    def get_pipeline(self, with_predictor=True):
        steps = list(zip(self._get_step_names(), self._get_default_estimators()))
        if not with_predictor and self._has_predictor():
            steps = steps[:-1]
        return clone(NamedParamPipeline(steps, memory=self.memory))

    def get_grid_search(self, **gs_params):
        # TODO make gs_params explicit
        grid = self.get_param_grid()
        if any(
            (hasattr(param, "rvs") and not isinstance(param, Sequence))
            for subgrid in grid
            for param in subgrid.values()
        ):
            raise ValueError(
                "Cannot get grid search if some of the choices are random. "
                "Use get_randomized_search() instead."
            )
        return GridSearchCV(self.get_pipeline(), grid, **gs_params)

    def get_randomized_search(self, **rs_params):
        # TODO make rs_params explicit
        grid = self.get_param_grid()
        return RandomizedSearchCV(self.get_pipeline(), grid, **rs_params)

    def truncated(self, before_step=None):
        if before_step is None:
            return self.truncated(1)
        if isinstance(before_step, str):
            names = self._get_step_names()
            if before_step not in names:
                raise ValueError(
                    f"{before_step!r} is not one of the step names: {names}."
                )
            idx = names.index(before_step)
            return self.truncated(idx)
        steps = self._steps[:before_step]
        if not steps:
            raise ValueError("Cannot truncate step 0 which separates X from y.")
        return self._with_prepared_steps(steps)

    def _transform_preview(self, sampling_method, n=None):
        if self.input_data is None:
            return None
        df = self._get_sampler(sampling_method)(n)
        pipeline = self.get_pipeline(False)
        if not pipeline.steps:
            return df, []
        y = _squeeze(s.select(df, self.y_cols))
        for step_name, transformer in pipeline.steps:
            if not _is_passthrough(transformer):
                try:
                    df = transformer.fit_transform(df, y)
                except Exception as e:
                    e_repr = "\n    ".join(traceback.format_exception_only(e))
                    raise ValueError(
                        f"Transformation failed at step '{step_name}'.\n"
                        f"Input data for this step:\n{df}\n"
                        f"Error message:\n    {e_repr}"
                    ) from e
        if _is_passthrough(pipeline.steps[-1][1]):
            return df, []
        return df, pipeline.steps[-1][1].created_outputs_

    def _check_n(self, n):
        if n is None:
            n = self.preview_sample_size
        elif n == -1:
            n = self.input_data_shape[0]
        return min(n, self.input_data_shape[0])

    def _random_sample(self, n):
        return sbd.sample(self.input_data, self._check_n(n), seed=self.random_seed)

    def _head_sample(self, n):
        return sbd.head(self.input_data, n=self._check_n(n))

    def _get_sampler(self, sampling_method):
        return {"head": self._head_sample, "random": self._random_sample}[
            sampling_method
        ]

    def sample(self, n=None, last_step_only=False, sampling_method="random"):
        if (transform_result := self._transform_preview(sampling_method, n=n)) is None:
            return None
        data, last_step_cols = transform_result
        if last_step_only:
            data = s.select(data, last_step_cols)
        return data

    def get_x(self):
        return s.select(self.input_data, s.all() - self.y_cols)

    def get_x_train(self):
        x = self.get_x()
        x_train, _ = train_test_split(
            x, random_state=self.random_seed, shuffle=self.shuffle_split
        )
        return x_train

    def get_x_test(self):
        x = self.get_x()
        _, x_test = train_test_split(
            x, random_state=self.random_seed, shuffle=self.shuffle_split
        )
        return x_test

    def get_y(self):
        y = _squeeze(s.select(self.input_data, self.y_cols))
        return y

    def get_y_train(self):
        y = self.get_y()
        y_train, _ = train_test_split(
            y, random_state=self.random_seed, shuffle=self.shuffle_split
        )
        return y_train

    def get_y_test(self):
        y = self.get_y()
        _, y_test = train_test_split(
            y, random_state=self.random_seed, shuffle=self.shuffle_split
        )
        return y_test

    def get_report(
        self, order_by=None, sampling_method="random", n=None, last_step_only=False
    ):
        if (transform_result := self._transform_preview(sampling_method, n=n)) is None:
            return None
        data, last_step_cols = transform_result
        if last_step_only:
            data = s.select(
                data, last_step_cols + ([] if order_by is None else [order_by])
            )
        else:
            column_filters = {
                "last_step_output": {
                    "display_name": "Modified by last step",
                    "columns": last_step_cols,
                },
                "~last_step_output": {
                    "display_name": "Not modified by last step",
                    "columns": (~s.cols(*last_step_cols)).expand(data),
                },
            }
        title = (
            f"Preview {'of last step output' if last_step_only else ''} on"
            f" {sbd.shape(data)[0]} rows"
        )
        return TableReport(
            data, order_by=order_by, column_filters=column_filters, title=title
        )

    def get_param_grid_description(self):
        return grid_description(self.get_param_grid())

    def get_pipeline_description(self):
        return _describe_pipeline(zip(self._get_step_names(), self._steps))

    def get_params_description(self, params):
        return params_description(params)

    def get_cv_results_description(self, fitted_gs, max_entries=8):
        out = io.StringIO()
        out.write("Best params:\n")
        out.write(f"    score: {fitted_gs.best_score_:.3g}\n")
        for line in io.StringIO(self.get_params_description(fitted_gs.best_params_)):
            out.write("    " + line)
        out.write("All combinations:\n")
        all_params = fitted_gs.cv_results_["params"]
        scores = fitted_gs.cv_results_["mean_test_score"]
        for entry_idx, (score, params) in enumerate(zip(scores, all_params)):
            if entry_idx == max_entries:
                remaining = len(scores) - max_entries
                out.write(
                    f"[ ... {remaining} more entries, set 'max_entries' to see more]"
                )
                break
            out.write(f"    - score: {score:.3g}\n")
            for line in io.StringIO(self.get_params_description(params)):
                out.write("      " + line)
        return out.getvalue()

    def get_cv_results_table(
        self, fitted_search, return_metadata=False, detailed=False
    ):
        import pandas as pd

        all_rows = []
        param_names = set()
        log_scale_columns = set()
        for params in fitted_search.cv_results_["params"]:
            row = {}
            for param_id, param in params.items():
                choice_name = param.in_choice or param_id
                value = param.name or param.value
                row[choice_name] = value
                param_names.add(choice_name)
                if getattr(param, "is_from_log_scale", False):
                    log_scale_columns.add(choice_name)
            all_rows.append(row)

        metadata = {"log_scale_columns": list(log_scale_columns)}
        all_ordered_param_names = _get_all_param_names(self.get_param_grid())
        ordered_param_names = [n for n in all_ordered_param_names if n in param_names]
        table = pd.DataFrame(all_rows, columns=ordered_param_names)
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
        table.insert(0, "mean_test_score", fitted_search.cv_results_["mean_test_score"])
        if detailed:
            for k in result_keys[1:]:
                if k in fitted_search.cv_results_:
                    table.insert(table.shape[1], k, fitted_search.cv_results_[k])
        table = table.sort_values("mean_test_score", ascending=False, ignore_index=True)
        return (table, metadata) if return_metadata else table

    def plot_parallel_coord(
        self, fitted_search, colorscale=DEFAULT_COLORSCALE, min_score=None
    ):
        cv_results, metadata = self.get_cv_results_table(
            fitted_search, return_metadata=True, detailed=True
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

    def __repr__(self):
        n_steps = len(self._steps)
        predictor_info = ""
        step_names = self._get_step_names()
        if self._has_predictor():
            n_steps -= 1
            predictor_info = " + predictor"
        step_descriptions = [f"{i}: {name}" for i, name in enumerate(step_names)]
        pipe_description = (
            f"<{self.__class__.__name__}: {n_steps} transformations{predictor_info}>"
            + (f"\nSteps:\n{', '.join(step_descriptions)}" if self._steps else "")
        )
        if self.input_data is None:
            return pipe_description
        try:
            data_repr = repr(self.sample())
            return f"{pipe_description}\nSample of transformed data:\n{data_repr}"
        except Exception as e:
            return (
                f"{pipe_description}\n{e}Note:\n    Use `.sample()` to trigger the"
                " error again and see the full traceback.\n    You can remove steps"
                " from the pipeline with `.truncated(step)`."
            )

    def apply(self, *args, **kwargs):
        return self.add(*args, **kwargs)

    def add(
        self,
        estimator,
        cols=s.all(),
        name=None,
        keep_original=False,
        rename_columns="{}",
        allow_reject=False,
    ):
        if self._has_predictor():
            pred_name = self._get_step_names()[-1]
            raise ValueError(
                f"This pipeline already has a final predictor: {pred_name!r}. "
                "Therefore we cannot add more steps. "
                "You can remove the final step with '.truncated(-1)'."
            )
        if isinstance(estimator, Choice):
            estimator = estimator.map_values(_check_passthrough)
        else:
            estimator = _check_passthrough(estimator)
        step = Step(
            estimator=estimator,
            cols=s.make_selector(cols),
            name=name,
            keep_original=keep_original,
            rename_columns=rename_columns,
            allow_reject=allow_reject,
        )
        return self._with_prepared_steps(self._steps + [step])

    def drop(self, cols, name=None):
        return self.add(Drop(), cols=cols, name=name)

    def select(self, cols, name=None):
        return self.drop(s.inv(cols), name=name)


def _describe_choice(choice, buf):
    buf.write("    choose estimator from:\n")
    dash = "        - "
    for outcome in choice.outcomes:
        if outcome.name is not None:
            name = f"{outcome.name} = "
        else:
            name = ""
        write_indented(f"{dash}{name}", f"{outcome.value!r}\n", buf)


def _describe_pipeline(named_steps):
    buf = io.StringIO()
    for name, step in named_steps:
        buf.write(f"{name}:\n")
        if isinstance(step.estimator, Optional):
            buf.write("    OPTIONAL STEP\n")
        buf.write(f"    cols: {step.cols}\n")
        if isinstance(step.estimator, Choice) and not isinstance(
            step.estimator, Optional
        ):
            _describe_choice(step.estimator, buf)
        else:
            write_indented(
                "    estimator: ",
                f"{unwrap_default(step.estimator)!r}\n",
                buf,
            )
    return buf.getvalue()


def _get_all_param_names(grid):
    names = {}
    for subgrid in grid:
        for k, v in subgrid.items():
            if v.name is not None:
                k = v.name
            names[k] = None
    return list(names)


# Aliases until we settle on a name:


class Recipe(PipeBuilder):
    pass


class Chain(PipeBuilder):
    pass
