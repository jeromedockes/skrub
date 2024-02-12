import inspect
import re

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.utils import discovery

from . import _dataframe as sbd
from . import selectors as s
from ._check_input import CheckInputDataFrame
from ._clean_null_strings import CleanNullStrings
from ._datetime_encoder import DatetimeColumnEncoder, DatetimeEncoder
from ._gap_encoder import GapEncoder
from ._minhash_encoder import MinHashEncoder
from ._pandas_convert_dtypes import PandasConvertDTypes
from ._table_vectorizer import TableVectorizer
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

    def use(self, *args, columns=s.all(), **kwargs):
        estimator = estimator_class(*args, **kwargs)
        return self.use(estimator, columns=columns)

    use.__name__ = method_name
    use.__qualname__ = f"{namespace_class.__name__}.{method_name}"
    use.__signature__ = inspect.signature(estimator_class.__init__)
    use.__doc__ = estimator_class.__doc__
    setattr(namespace_class, method_name, use)


def _add_estimators_as_methods(cls):
    for _, estimator_class in discovery.all_estimators() + _SKRUB_TRANSFORMERS:
        _add_method(cls, estimator_class)
    return cls


@_add_estimators_as_methods
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
        self.steps = []
        self._has_predictor = False

    def use(self, estimator, columns=s.all()):
        if self._has_predictor:
            raise ValueError(
                "pipe already has a final predictor: "
                f"{self.steps[-1].__class__.__name__}. "
                "Cannot append more transformation steps."
            )
        if hasattr(estimator, "predict"):
            self._has_predictor = True
            self.steps.append(estimator)
        else:
            assert hasattr(estimator, "transform")
            self.steps.append(
                s.make_selector(columns).use(estimator, n_jobs=self.n_jobs)
            )
        return self

    def get_pipeline(self, with_predictor=True):
        steps = self.steps
        if not with_predictor and self._has_predictor:
            steps = steps[:-1]
        return clone(make_pipeline(*steps))

    def sample(self, n=None):
        if n is None:
            n = self.preview_sample_size
        if self.input_data is None:
            return None
        sample_data = sbd.sample(
            self.input_data, min(n, self.input_data_shape[0]), seed=self.random_seed
        )
        pipeline = self.get_pipeline(False)
        if not pipeline.steps:
            return sample_data
        return pipeline.fit_transform(sample_data)

    def __repr__(self):
        n_steps = len(self.steps)
        predictor_info = ""
        if self._has_predictor:
            n_steps -= 1
            predictor_info = f" + {self.steps[-1].__class__.__name__}"
        pipe_description = (
            f"<{self.__class__.__name__}: {n_steps} transformations{predictor_info}>"
        )
        if self.input_data is None:
            return pipe_description
        data_repr = repr(self.sample())
        return f"{pipe_description}\nSample of transformed data:\n{data_repr}"
