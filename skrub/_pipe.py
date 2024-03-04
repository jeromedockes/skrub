import io
import itertools

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from . import _dataframe as sbd
from . import selectors as s
from ._add_estimator_methods import add_estimator_methods, camel_to_snake
from ._choice import (
    Choice,
    choose,
    expand_grid,
    find_param_choice,
    first,
    grid_description,
    set_params_to_first,
)


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
        or getattr(first(step), "name_", None)
        or camel_to_snake(first(step).estimator_.__class__.__name__)
    )


def _to_step(obj):
    if isinstance(obj, Choice):
        return choose(*map(_to_step, obj.options_)).name(obj.name_)
    if hasattr(obj, "estimator_"):
        return obj
    return s.all().use(obj)


def _to_estimator(step, n_jobs):
    if isinstance(step, Choice):
        return choose(*(_to_estimator(opt, n_jobs) for opt in step.options_)).name(
            step.name_
        )
    if hasattr(step.estimator_, "predict"):
        return step.estimator_
    return step.cols_.make_transformer(step.estimator_, n_jobs=n_jobs)


def _contains_choice(estimator):
    return isinstance(estimator, Choice) or (find_param_choice(estimator) is not None)


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

    def _with_steps(self, steps):
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
        return hasattr(first(self._steps[-1]).estimator_, "predict")

    def _get_step_names(self):
        suggested_names = [_get_name(step) for step in self._steps]
        return _pick_names(suggested_names)

    def _get_estimators(self):
        return [_to_estimator(step, self.n_jobs) for step in self._steps]

    def _get_default_estimators(self):
        return list(map(set_params_to_first, self._get_estimators()))

    def _get_param_grid(self):
        grid = {
            name: estimator
            for (name, estimator) in zip(self._get_step_names(), self._get_estimators())
            if _contains_choice(estimator)
        }
        return expand_grid(grid)

    def _get_pipeline(self, with_predictor=True):
        steps = list(zip(self._get_step_names(), self._get_default_estimators()))
        if not with_predictor and self._has_predictor():
            steps = steps[:-1]
        return clone(Pipeline(steps))

    def _get_grid_search(self):
        grid = self._get_param_grid()
        return GridSearchCV(self._get_pipeline(), grid)

    def chain(self, *steps):
        return self._with_steps(self._steps + list(map(_to_step, steps)))

    def pop(self):
        if not self._steps:
            return None
        estimator = _to_estimator(self._steps[-1], self.n_jobs)
        self._steps = self._steps[:-1]
        return estimator

    def remove(self, step=-1):
        if isinstance(step, str):
            step = self._get_step_names().index(step)
        del self._steps[step]
        return self

    @property
    def grid_search(self):
        return self._get_grid_search()

    @property
    def pipeline(self):
        return self._get_pipeline()

    def sample(self, n=None):
        if n is None:
            n = self.preview_sample_size
        if self.input_data is None:
            return None
        sample_data = sbd.sample(
            self.input_data, min(n, self.input_data_shape[0]), seed=self.random_seed
        )
        pipeline = self._get_pipeline(False)
        if not pipeline.steps:
            return sample_data
        return pipeline.fit_transform(sample_data)

    @property
    def param_grid_description(self):
        return grid_description(self._get_param_grid())

    @property
    def pipeline_description(self):
        return _describe_pipeline(zip(self._get_step_names(), self._steps))

    def __repr__(self):
        n_steps = len(self._steps)
        predictor_info = ""
        if self._has_predictor():
            n_steps -= 1
            predictor_info = (
                f" + {first(self._steps[-1]).estimator_.__class__.__name__}"
            )
        step_descriptions = [
            f"{i}: {name}" for i, name in enumerate(self._get_step_names())
        ]
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
                f"{pipe_description}\nTransform failed:\n    {type(e).__name__}:"
                f" {e}\nNote:\nYou can inspect pipeline steps with `.steps` or remove"
                " steps with `.pop()` or `remove()`."
            )

    # Alternative API 1
    def use(self, estimator, cols=s.all(), name=None):
        return self.chain(cols.use(estimator).name(name))

    # Alternative API 2
    def cols(self, selector):
        return StepCols(self, selector)


# Alternative API 2
@add_estimator_methods
class StepCols:
    def __init__(self, pipe, cols):
        self.pipe_ = pipe
        self.cols_ = s.make_selector(cols)

    def use(self, estimator):
        return self.pipe_.chain(self.cols_.use(estimator))

    def __repr__(self):
        return f"<TODO columns: {self.cols_}>"


def _describe_choice(name, choice, buf):
    buf.write(f"{name}:\n")
    buf.write("    choose:\n")
    dash = "        - "
    indent = "          "
    for opt in choice.options_:
        buf.write(f"{dash}cols: {opt.cols_}\n")
        buf.write(f"{indent}estimator: {opt.estimator_}\n")


def _describe_step(name, step, buf):
    buf.write(f"{name}:\n")
    buf.write(f"    cols: {step.cols_}\n")
    buf.write(f"    estimator: {step.estimator_}\n")


def _describe_pipeline(named_steps):
    buf = io.StringIO()
    for name, step in named_steps:
        if isinstance(step, Choice):
            _describe_choice(name, step, buf)
        else:
            _describe_step(name, step, buf)
    return buf.getvalue()
