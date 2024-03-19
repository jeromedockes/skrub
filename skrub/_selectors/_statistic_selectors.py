from .. import _dataframe as sbd
from ._base import column_selector


@column_selector
def cardinality_below(column, threshold):
    try:
        return sbd.n_unique(column) < threshold
    except Exception:
        # n_unique can fail for example for polars columns with dtype Object
        return False
