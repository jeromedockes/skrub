"""Helpers for selecting columns in a dataframe.

TODO

>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )
>>> from skrub import selectors as s
>>> s.select(df, ["ID", "kind"])
   ID kind
0   4   A4
1   3   A3
>>> s.select(df, s.all())
   height_mm  width_mm kind  ID
0      297.0     210.0   A4   4
1      420.0     297.0   A3   3
>>> s.select(df, s.numeric() - "ID")
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0
>>> s.select(df, s.string() | ["width_mm", "ID"])
   width_mm kind  ID
0     210.0   A4   4
1     297.0   A3   3
>>> s.select(df, s.numeric() - s.glob("*_mm"))
   ID
0   4
1   3
>>> non_mm = s.numeric() - s.glob("*_mm")
>>> non_mm
(numeric() - glob('*_mm'))
>>> s.select(df, non_mm)
   ID
0   4
1   3
"""

from . import _selectors
from ._base import Selector, all, cols, filter, filter_names, inv, make_selector, select
from ._selectors import *  # noqa: F403,F401

__all__ = [
    "Selector",
    "all",
    "cols",
    "filter",
    "filter_names",
    "inv",
    "make_selector",
    "select",
]
__all__ += _selectors.__all__
