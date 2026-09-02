"""
functions meant to be cached with joblib.

They are in their own module so the cache is less likely to be invalidated due
to the line number of the function definition changing.
"""


def _call_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    result = getattr(estimator, method_name)(*args, **kwargs)
    return estimator, result


def _call_non_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    return getattr(estimator, method_name)(*args, **kwargs)
