from .. import _dataframe as sbd
from ._base import Filter


def cardinality_below(threshold):
    # on_error='reject' because n_unique can fail for example for polars columns
    # with dtype Object
    return Filter(
        lambda c: sbd.n_unique(c) < threshold,
        on_error="reject",
        name="cardinality_below",
        args=(threshold,),
    )
