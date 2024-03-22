import io
import itertools
import traceback

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from . import _dataframe as sbd
from . import selectors as s
from ._add_estimator_methods import camel_to_snake
from ._tuning import (
    Choice,
    NumericChoice,
    Optional,
    choose,
    choose_float,
    choose_int,
    contains_choice,
    expand_grid,
    grid_description,
    optional,
    params_description,
    set_params_to_first,
    unwrap,
    unwrap_first,
    write_indented,
)

__all__ = ["Pipe", "choose", "optional", "choose_float", "choose_int"]


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
        getattr(step, "name_", None)
        or getattr(step.estimator_, "name_", None)
        or camel_to_snake(unwrap_first(step.estimator_).__class__.__name__)
    )


def _to_step(obj):
    if not hasattr(obj, "estimator_"):
        obj = s.all().use(obj)
    if isinstance(obj.estimator_, Choice):
        obj = obj._with_params(estimator=obj.estimator_.map_values(_check_passthrough))
    else:
        obj = obj._with_params(estimator=_check_passthrough(obj.estimator_))
    return obj


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
    estimator = step.estimator_
    if isinstance(estimator, Choice):
        return estimator.map_values(lambda v: _check_estimator(v, step, n_jobs))
    return _check_estimator(estimator, step, n_jobs)


class Pipe:
    def __init__(
        self, input_data=None, n_jobs=None, preview_sample_size=200, random_seed=0
    ):
        self.input_data = input_data
        if input_data is None:
            self.input_data_shape = None
        else:
            self.input_data_shape = sbd.shape(input_data)
        self.n_jobs = n_jobs
        self.preview_sample_size = preview_sample_size
        self.random_seed = random_seed
        self._steps = []

    def _with_prepared_steps(self, steps):
        new = Pipe(
            input_data=self.input_data,
            n_jobs=self.n_jobs,
            preview_sample_size=self.preview_sample_size,
            random_seed=self.random_seed,
        )
        new._steps = steps
        return new

    def _has_predictor(self):
        if not self._steps:
            return False
        return hasattr(unwrap_first(self._steps[-1].estimator_), "predict")

    def _get_step_names(self):
        suggested_names = [_get_name(step) for step in self._steps]
        return _pick_names(suggested_names)

    def _get_estimators(self):
        return [_to_estimator(step, self.n_jobs) for step in self._steps]

    def _get_default_estimators(self):
        return [
            set_params_to_first(unwrap_first(estimator))
            for estimator in self._get_estimators()
        ]

    def _get_param_grid(self):
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
        return clone(NamedParamPipeline(steps))

    def get_grid_search(self, **gs_params):
        # TODO make gs_params explicit
        grid = self._get_param_grid()
        if any(
            isinstance(param, NumericChoice)
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
        grid = self._get_param_grid()
        return RandomizedSearchCV(self.get_pipeline(), grid, **rs_params)

    def chain(self, *steps):
        if self._has_predictor():
            pred_name = self._get_step_names()[-1]
            raise ValueError(
                f"This pipeline already has a final predictor: {pred_name!r}. "
                "Therefore we cannot add more steps. "
                "You can remove the final step with '.pop()'."
            )
        return self._with_prepared_steps(self._steps + list(map(_to_step, steps)))

    def pop(self):
        if not self._steps:
            return None
        estimator = _to_estimator(self._steps[-1], self.n_jobs)
        self._steps = self._steps[:-1]
        return estimator

    def _transform_preview(self, sampling_method, n=None):
        if self.input_data is None:
            return None
        n = self.preview_sample_size if n is None else n
        df = self._get_sampler(sampling_method)(n)
        pipeline = self.get_pipeline(False)
        if not pipeline.steps:
            return df, []
        for step_name, transformer in pipeline.steps:
            try:
                df = transformer.fit_transform(df)
            except Exception as e:
                e_repr = "\n    ".join(traceback.format_exception_only(e))
                raise RuntimeError(
                    f"Transformation failed at step '{step_name}'.\n"
                    f"Input data for this step:\n{df}\n"
                    f"Error message:\n    {e_repr}"
                ) from e
        return df, pipeline.steps[-1][1].created_outputs_

    def _random_sample(self, n):
        return sbd.sample(
            self.input_data, min(n, self.input_data_shape[0]), seed=self.random_seed
        )

    def _head_sample(self, n):
        return sbd.head(self.input_data, n=n)

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

    def get_skrubview_report(
        self, order_by=None, sampling_method="random", n=None, last_step_only=False
    ):
        try:
            import skrubview
        except ImportError:
            print("Please install skrubview")
            return None

        if (transform_result := self._transform_preview(sampling_method, n=n)) is None:
            return None
        data, last_step_cols = transform_result
        if last_step_only:
            data = s.select(
                data, last_step_cols + ([] if order_by is None else [order_by])
            )
        return skrubview.Report(data, order_by=order_by)

    def get_param_grid_description(self):
        return grid_description(self._get_param_grid())

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

    def get_cv_results_table(self, fitted_gs, convert_log_scale=False):
        import pandas as pd

        all_params = fitted_gs.cv_results_["params"]
        mean_test_scores = fitted_gs.cv_results_["mean_test_score"]
        std_test_scores = fitted_gs.cv_results_["std_test_score"]
        mean_fit_time = fitted_gs.cv_results_["mean_fit_time"]
        all_rows = []
        param_names = set()
        for score, std, time, params in zip(
            mean_test_scores, std_test_scores, mean_fit_time, all_params
        ):
            row = {"mean_score": score, "fit_time": time, "std_score": std}
            for param_id, param in params.items():
                choice_name = param.in_choice_ or param_id
                value = param.name_ or param.value_
                if convert_log_scale and getattr(param, "is_from_log_scale_", False):
                    choice_name = f"log10({choice_name})"
                    value = np.log10(value)
                row[choice_name] = value
                param_names.add(choice_name)
            all_rows.append(row)
        all_ordered_param_names = _get_all_param_names(self._get_param_grid())
        all_ordered_param_names = itertools.chain(
            *((n, f"log10({n})") for n in all_ordered_param_names)
        )
        ordered_param_names = [n for n in all_ordered_param_names if n in param_names]
        cols = ["mean_score"] + ordered_param_names + ["fit_time", "std_score"]
        table = pd.DataFrame(all_rows, columns=cols).sort_values(
            "mean_score", ascending=False, ignore_index=True
        )
        return table

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
                f"{pipe_description}\n"
                f"{e}Note:\n"
                "    Use `.sample()` to trigger the error again "
                "and see the full traceback.\n"
                "    You can remove steps from the pipeline with `.pop()`."
            )

    # Alternative API 1
    def use(
        self,
        estimator,
        cols=s.all(),
        name=None,
        keep_original=False,
        rename_columns="{}",
    ):
        return self.chain(
            s.make_selector(cols)
            .use(estimator)
            .name(name)
            .keep_original(keep_original)
            .rename_columns(rename_columns)
        )


def _describe_choice(choice, buf):
    buf.write("    choose estimator from:\n")
    dash = "        - "
    for outcome in choice.outcomes_:
        if outcome.name_ is not None:
            name = f"{outcome.name_} = "
        else:
            name = ""
        write_indented(f"{dash}{name}", f"{outcome.value_!r}\n", buf)


def _describe_pipeline(named_steps):
    buf = io.StringIO()
    for name, step in named_steps:
        buf.write(f"{name}:\n")
        if isinstance(step.estimator_, Optional):
            buf.write("    OPTIONAL STEP\n")
        buf.write(f"    cols: {step.cols_}\n")
        if isinstance(step.estimator_, Choice) and not isinstance(
            step.estimator_, Optional
        ):
            _describe_choice(step.estimator_, buf)
        else:
            write_indented(
                "    estimator: ",
                f"{unwrap_first(step.estimator_)!r}\n",
                buf,
            )
    return buf.getvalue()


def _get_all_param_names(grid):
    names = {}
    for subgrid in grid:
        for k, v in subgrid.items():
            if v.name_ is not None:
                k = v.name_
            names[k] = None
    return list(names)
