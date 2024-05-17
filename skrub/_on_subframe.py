from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors
from ._join_utils import pick_column_names
from ._utils import renaming_func


class OnSubFrame(TransformerMixin, BaseEstimator):
    """Apply a transformer to part of a dataframe.

    A subset of the dataframe is selected and passed to the transformer (as a
    single input). This is different from ``Map`` which fits a separate clone
    of the transformer to each selected column independently.

    Parameters
    ----------
    transformer : scikit-learn Transformer
        The transformer to apply to the selected columns. ``fit_transform`` and
        ``transform`` must return a DataFrame. The resulting dataframe will
        appear as the last columns of the output dataframe. Unselected columns
        will appear unchanged in the output.

    cols : str, sequence of str, or skrub selector, optional
        The columns to attempt to transform. Columns outside of this selection
        will be passed through unchanged, without calling ``fit_transform`` on
        them. The default is transform all columns.

    keep_original : bool, default=False
        If ``True``, the original columns are preserved in the output. If the
        transformer produces a column with the same name, the transformation
        result is renamed so that both columns can appear in the output. If
        ``False``, only the transformer's output is included in the result, not
        the original columns. In all cases columns not selected by ``cols`` are
        passed through.

    rename_columns : str, default='{}'
        Format strings applied to all transformation ouput column names. For
        example pass ``'transformed_{}'`` to prepend ``'transformed_'`` to all
        output column names. The default value does not modify the names.
        Renaming is not applied to columns not selected by ``cols``.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.

    used_inputs_ : list of str
        The names of columns that were transformed.

    all_outputs_ : list of str
        All column names in the output dataframe.

    created_outputs_ : list of str
        The names of columns in the output dataframe that were created by the
        fitted transformer.

    transformer_ : Transformer
        The fitted transformer.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.eye(4) * np.logspace(0, 3, 4), columns=list("abcd"))
    >>> df
         a     b      c       d
    0  1.0   0.0    0.0     0.0
    1  0.0  10.0    0.0     0.0
    2  0.0   0.0  100.0     0.0
    3  0.0   0.0    0.0  1000.0
    >>> from sklearn.decomposition import PCA
    >>> from skrub._on_subframe import OnSubFrame
    >>> OnSubFrame(PCA(n_components=2)).fit_transform(df).round(2)
         pca0   pca1
    0 -249.01 -33.18
    1 -249.04 -33.68
    2 -252.37  66.64
    3  750.42   0.22

    We can restrict the transformer to a subset of columns:

    >>> pca = OnSubFrame(PCA(n_components=2), cols=["a", "b"])
    >>> pca.fit_transform(df).round(2)
           c       d  pca0  pca1
    0    0.0     0.0 -2.52  0.67
    1    0.0     0.0  7.50  0.00
    2  100.0     0.0 -2.49 -0.33
    3    0.0  1000.0 -2.49 -0.33
    >>> pca.used_inputs_
    ['a', 'b']
    >>> pca.created_outputs_
    ['pca0', 'pca1']
    >>> pca.transformer_
    PCA(n_components=2)

    It is possible to rename the output columns:

    >>> pca = OnSubFrame(PCA(n_components=2), cols=["a", "b"], rename_columns='my_tag-{}')
    >>> pca.fit_transform(df).round(2)
           c       d  my_tag-pca0  my_tag-pca1
    0    0.0     0.0        -2.52         0.67
    1    0.0     0.0         7.50         0.00
    2  100.0     0.0        -2.49        -0.33
    3    0.0  1000.0        -2.49        -0.33

    We can also force preserving the original columns in the output:

    >>> pca = OnSubFrame(PCA(n_components=2), cols=["a", "b"], keep_original=True)
    >>> pca.fit_transform(df).round(2)
         a     b      c       d  pca0  pca1
    0  1.0   0.0    0.0     0.0 -2.52  0.67
    1  0.0  10.0    0.0     0.0  7.50  0.00
    2  0.0   0.0  100.0     0.0 -2.49 -0.33
    3  0.0   0.0    0.0  1000.0 -2.49 -0.33
    """

    def __init__(
        self,
        transformer,
        cols=_selectors.all(),
        keep_original=False,
        rename_columns="{}",
    ):
        self.transformer = transformer
        self.cols = cols
        self.keep_original = keep_original
        self.rename_columns = rename_columns

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        self.all_inputs_ = sbd.column_names(X)
        self._columns = _selectors.make_selector(self.cols).expand(X)
        to_transform = _selectors.select(X, self._columns)
        if self.keep_original:
            passthrough = X
        else:
            passthrough = _selectors.select(X, _selectors.inv(self._columns))
        passthrough_names = sbd.column_names(passthrough)
        if self._columns:
            self.transformer_ = clone(self.transformer)
            if hasattr(self.transformer_, "set_output"):
                df_module_name = sbd.dataframe_module_name(X)
                self.transformer_.set_output(transform=df_module_name)
            transformed = self.transformer_.fit_transform(to_transform, y)
            suggested_names = sbd.column_names(transformed)
            suggested_names = list(
                map(renaming_func(self.rename_columns), suggested_names)
            )
            self._transformed_output_names = pick_column_names(
                suggested_names, forbidden_names=passthrough_names
            )
            transformed = sbd.set_column_names(
                transformed, self._transformed_output_names
            )
            result = sbd.concat_horizontal(passthrough, transformed)
        else:
            self.transformer_ = None
            result = passthrough
            self._transformed_output_names = []
        self.used_inputs_ = self._columns
        self.created_outputs_ = self._transformed_output_names
        self.all_outputs_ = passthrough_names + self._transformed_output_names
        self.input_to_outputs_ = {c: self.created_outputs_ for c in self.used_inputs_}
        # for sklearn
        self.feature_names_in_ = self.all_inputs_
        self.n_features_in_ = len(self.all_inputs_)

        return result

    def transform(self, X):
        # do the selection even if self._columns is empty to raise if X doesn't
        # have the right columns
        to_transform = _selectors.select(X, self._columns)
        if self.keep_original:
            passthrough = X
        else:
            passthrough = _selectors.select(X, _selectors.inv(self._columns))
        if not self._columns:
            return passthrough
        transformed = self.transformer_.transform(to_transform)
        transformed = sbd.set_column_names(transformed, self._transformed_output_names)
        return sbd.concat_horizontal(passthrough, transformed)

    # set_output api compatibility

    def get_feature_names_out(self):
        return self.all_outputs_
