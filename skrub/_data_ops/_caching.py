"""
Helper to cache estimator functions according to the config.
"""

import pickle

import joblib

from .. import _config

# The functions meant to be cached. They are at the top because joblib
# invalidates the cache when the line number changes.


def _call_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    result = getattr(estimator, method_name)(*args, **kwargs)
    return estimator, result


def _call_non_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    return getattr(estimator, method_name)(*args, **kwargs)


class Memory:
    def __init__(self):
        self.cache_dir = None
        self.memory = None
        self.cached_func = {}

    def _check_cache_dir(self):
        cache_dir = _config.get_config()["cache_dir"]
        if cache_dir == self.cache_dir:
            return
        self.cached_func = {}
        self.memory = joblib.Memory(cache_dir, verbose=0)
        self.cache_dir = cache_dir

    def has_memory(self):
        self._check_cache_dir()
        return self.memory is not None

    def cache(self, func, ignore=()):
        self._check_cache_dir()
        if self.memory is None:
            return func
        key = (func, ignore)
        try:
            return self.cached_func[key]
        except KeyError:
            pass
        result = self.memory.cache(func, ignore=ignore)
        self.cached_func[key] = result
        return result

    def call_fitting_method(self, estimator, method_name, args, kwargs):
        if not self.has_memory():
            result = getattr(estimator, method_name)(*args, **kwargs)
            return estimator, result, None
        try:
            estimator_id = joblib.hash((estimator, method_name, args, kwargs))
            estimator, result = self.cache(
                _call_fitting_method,
                ignore=("estimator", "method_name", "args", "kwargs"),
            )(estimator, method_name, args, kwargs, estimator_id)
            return estimator, result, estimator_id
        except pickle.PicklingError:
            pass
        # Fall back to non-cached call if arguments cannot be serialized
        result = getattr(estimator, method_name)(*args, **kwargs)
        return estimator, result, None

    def call_non_fitting_method(
        self, estimator, method_name, args, kwargs, estimator_id
    ):
        if not self.has_memory() or estimator_id is None:
            return getattr(estimator, method_name)(*args, **kwargs)
        try:
            return self.cache(_call_non_fitting_method, ignore=("estimator",))(
                estimator, method_name, args, kwargs, estimator_id
            )
        except pickle.PicklingError:
            pass
        # Fall back to non-cached call if arguments cannot be serialized
        return getattr(estimator, method_name)(*args, **kwargs)
