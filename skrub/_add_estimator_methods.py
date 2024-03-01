import inspect
import re

from sklearn.utils import discovery

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


def add_estimators_as_methods(cls):
    sklearn_estimators = [
        (name, cls) for name, cls in discovery.all_estimators() if name != "Pipeline"
    ]
    for _, estimator_class in sklearn_estimators + _SKRUB_TRANSFORMERS:
        _add_method(cls, estimator_class)
    return cls
