import inspect
import itertools
import re

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import discovery

from . import _dataframe as sbd
from . import selectors as s
from ._check_input import CheckInputDataFrame
from ._clean_null_strings import CleanNullStrings
from ._datetime_encoder import DatetimeColumnEncoder, DatetimeEncoder
from ._gap_encoder import GapEncoder
from ._minhash_encoder import MinHashEncoder
from ._pandas_convert_dtypes import PandasConvertDTypes
from ._table_vectorizer import Drop, TableVectorizer
from ._to_categorical import ToCategorical
from ._to_datetime import ToDatetime
from ._to_float import ToFloat32
from ._to_numeric import ToNumeric

_SKRUB_TRANSFORMERS = [
    (c.__name__, c)
    for c in [
        CheckInputDataFrame,
        CleanNullStrings,
        DatetimeColumnEncoder,
        DatetimeEncoder,
        GapEncoder,
        MinHashEncoder,
        PandasConvertDTypes,
        ToCategorical,
        ToDatetime,
        ToNumeric,
        ToFloat32,
        TableVectorizer,
        Drop,
    ]
]


def _camel_to_snake(name):
    name = re.sub(
        r"(.)([A-Z][a-z0-9]+)",
        r"\1_\2",
        name,
    )
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def _add_method(namespace_class, estimator_class, method_name=None):
    if method_name is None:
        method_name = _camel_to_snake(estimator_class.__name__)

    def use(self, *args, **kwargs):
        estimator = estimator_class(*args, **kwargs)
        return self.use(estimator)

    use.__name__ = method_name
    use.__qualname__ = f"{namespace_class.__name__}.{method_name}"
    use.__signature__ = inspect.signature(estimator_class.__init__)
    use.__doc__ = estimator_class.__doc__
    setattr(namespace_class, method_name, use)


def _add_estimators_as_methods(cls):
    sklearn_estimators = [
        (name, cls) for name, cls in discovery.all_estimators() if name != "Pipeline"
    ]
    for _, estimator_class in sklearn_estimators + _SKRUB_TRANSFORMERS:
        _add_method(cls, estimator_class)
    return cls


class _Step:
    def __init__(self, estimator):
        self.estimator = estimator
        self.step_name = None
        self.param_grid = {}
        self.on_cols = s.all()


class StepConfig:
    def __init__(self, pipeline, step):
        self.pipeline = pipeline
        self.step = step

    def step_name(self, name):
        self.step.step_name = name
        return self

    def param_grid(self, **grid):
        self.step.param_grid = grid
        return self

    def on_cols(self, cols):
        self.step.on_cols = cols
        return self

    def replace(self, other=None):
        self.pipeline._steps.remove(self.step)
        if other is None:
            other = _get_name(self.step)
        self.pipeline._replace_step(self.step, other)
        return self

    def replace_last(self):
        return self.replace(-1)

    def insert(self, idx):
        self.pipeline._steps.remove(self.step)
        self.pipeline._insert_step(idx, self.step)
        return self

    def __repr__(self):
        return (
            f"<StepConfig for {self.step.estimator.__class__.__name__}>\n"
            f"Full pipeline:\n{self.pipeline!r}"
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
    return step.step_name or _camel_to_snake(step.estimator.__class__.__name__)


@_add_estimators_as_methods
class TransformedData:
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

    def _has_predictor(self):
        if not self._steps:
            return False
        return hasattr(self._steps[-1].estimator, "predict")

    def use(self, estimator):
        step = _Step(estimator)
        self._steps.append(step)
        return StepConfig(self, step)

    @property
    def step_names(self):
        suggested_names = [_get_name(step) for step in self._steps]
        return _pick_names(suggested_names)

    def _prepare_steps(self):
        prepared_steps = []
        param_grid = {}
        for name, step in zip(self.step_names, self._steps):
            if hasattr(step.estimator, "predict"):
                prepared_steps.append((name, step.estimator))
                grid_name = name
            else:
                prepared_steps.append(
                    (
                        name,
                        s.make_selector(step.on_cols).use(
                            step.estimator, n_jobs=self.n_jobs
                        ),
                    )
                )
                grid_name = f"{name}__transformer"
            for param_name, param_values in step.param_grid.items():
                param_grid[f"{grid_name}__{param_name}"] = param_values
        return prepared_steps, param_grid

    def pop(self):
        if not self._steps:
            return None
        step = self._prepare_steps()[0][-1]
        self._steps = self._steps[:-1]
        return step

    def remove(self, step=-1):
        if isinstance(step, str):
            step = self.step_names.index(step)
        del self._steps[step]
        return self

    def _replace_step(self, new, step=-1):
        if isinstance(step, str):
            step = self.step_names.index(step)
        self._steps[step] = new
        return self

    def _insert_step(self, idx, new):
        if not isinstance(idx, int):
            raise TypeError("idx must be an int")
        self._steps.insert(idx, new)
        return self

    @property
    def steps(self):
        return self._prepare_steps()[0]

    @property
    def param_grid(self):
        return self._prepare_steps()[1]

    @property
    def grid_search(self):
        steps, grid = self._prepare_steps()
        return GridSearchCV(clone(Pipeline(steps)), grid)

    def _get_pipeline(self, with_predictor=True):
        steps, _ = self._prepare_steps()
        if not with_predictor and self._has_predictor():
            steps = steps[:-1]
        return clone(Pipeline(steps))

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

    def __repr__(self):
        n_steps = len(self._steps)
        predictor_info = ""
        if self._has_predictor():
            n_steps -= 1
            predictor_info = f" + {self._steps[-1].estimator.__class__.__name__}"
        step_descriptions = [f"{i}: {name}" for i, name in enumerate(self.step_names)]
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
                " steps with `.pop()` or `remove()`.\nInstead of adding a step you can"
                " also replace one, for example:\n`pipe.to_datetime().replace()`"
                " instead of `pipe.to_datetime()`."
            )
