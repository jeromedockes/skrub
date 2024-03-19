from .. import _dataframe as sbd
from ._base import Selector


class CardinalityBelow(Selector):
    def __init__(self, threshold):
        self.threshold = threshold

    def matches(self, col):
        try:
            n_unique = sbd.n_unique(col)
        except Exception:
            # n_unique can fail for example for polars columns with dtype Object
            return False
        return n_unique < self.threshold

    def __repr__(self):
        return f"cardinality_below({self.threshold})"


def cardinality_below(threshold):
    return CardinalityBelow(threshold)
