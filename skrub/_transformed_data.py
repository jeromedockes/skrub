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
from ._datetime_encoder import DatetimeEncoder, EncodeDatetime
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
        EncodeDatetime,
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
    def __init__(self, estimator, cols=s.all(), param_grid=None, step_name=None):
        self.estimator = estimator
        self.cols = cols
        self.param_grid = {} if param_grid is None else param_grid
        self.step_name = step_name


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

    def _with_added_step(self, step):
        new = TransformedData(
            input_data=self.input_data,
            n_jobs=self.n_jobs,
            preview_sample_size=self.preview_sample_size,
            random_seed=self.random_seed,
        )
        new._steps = self._steps + [step]
        return new

    def _has_predictor(self):
        if not self._steps:
            return False
        return hasattr(self._steps[-1].estimator, "predict")

    def use(self, estimator, cols=s.all(), param_grid=None, step_name=None):
        step = _Step(estimator, cols=cols, param_grid=param_grid, step_name=step_name)
        return self._with_added_step(step)

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
                        s.make_selector(step.cols).use(
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
                " steps with `.pop()` or `remove()`."
            )
