"""
skrub: Prepping tables for machine learning.
"""

from pathlib import Path as _Path

from . import _selectors as selectors
from ._agg_joiner import AggJoiner, AggTarget
from ._check_dependencies import check_dependencies
from ._column_associations import column_associations
from ._datetime_encoder import DatetimeEncoder
from ._deduplicate import compute_ngram_distance, deduplicate
from ._expressions import (
    X,
    cross_validate,
    deferred,
    deferred_optional,
    if_else,
    value,
    var,
    y,
)
from ._fuzzy_join import fuzzy_join
from ._gap_encoder import GapEncoder
from ._interpolation_joiner import InterpolationJoiner
from ._joiner import Joiner
from ._minhash_encoder import MinHashEncoder
from ._multi_agg_joiner import MultiAggJoiner
from ._reporting import TableReport, patch_display, unpatch_display
from ._select_cols import Drop, DropCols, SelectCols
from ._similarity_encoder import SimilarityEncoder
from ._string_encoder import StringEncoder
from ._table_vectorizer import TableVectorizer
from ._tabular_learner import tabular_learner
from ._text_encoder import TextEncoder
from ._to_categorical import ToCategorical
from ._to_datetime import ToDatetime, to_datetime
from ._tuning import (
    choose_bool,
    choose_float,
    choose_from,
    choose_int,
    optional,
)

check_dependencies()

with open(_Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()


__all__ = [
    "var",
    "X",
    "y",
    "value",
    "deferred",
    "deferred_optional",
    "cross_validate",
    "if_else",
    "TableReport",
    "patch_display",
    "unpatch_display",
    "tabular_learner",
    "DatetimeEncoder",
    "ToDatetime",
    "Joiner",
    "fuzzy_join",
    "GapEncoder",
    "InterpolationJoiner",
    "MinHashEncoder",
    "SimilarityEncoder",
    "TableVectorizer",
    "deduplicate",
    "compute_ngram_distance",
    "ToCategorical",
    "to_datetime",
    "AggJoiner",
    "MultiAggJoiner",
    "AggTarget",
    "SelectCols",
    "DropCols",
    "Drop",
    "selectors",
    "Recipe",
    "choose_from",
    "optional",
    "choose_float",
    "choose_int",
    "choose_bool",
    "selectors",
    "TextEncoder",
    "StringEncoder",
    "column_associations",
]
