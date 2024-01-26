from sklearn.base import BaseEstimator, TransformerMixin

from . import _dataframe as sb
from ._dataframe import asdfapi


class CheckInputDataFrame(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        del y
        module_name = sb.dataframe_module_name(X)
        if module_name is None:
            raise TypeError(
                "Only pandas DataFrames and polars DataFrames and LazyFrames are"
                f" supported. Cannot handle X of type: {type(X)}"
            )
        self.module_name_ = module_name
        # TODO check schema (including dtypes) not just names.
        # Need to decide how strict we should be about types
        self.column_names_ = asdfapi(X).column_names
        return self

    def transform(self, X):
        module_name = sb.dataframe_module_name(X)
        if module_name is None:
            raise TypeError(
                "Only pandas DataFrames and polars DataFrames and LazyFrames are"
                f" supported. Cannot handle X of type: {type(X)}"
            )
        if module_name != self.module_name_:
            # TODO should this be a warning instead?
            raise TypeError(
                f"Pipeline was fitted to a {self.module_name_} dataframe "
                f"but is being applied to a {module_name} dataframe. "
                "This is likely to produce errors and is not supported."
            )
        column_names = asdfapi(X).column_names
        if column_names == self.column_names_:
            return X
        import difflib

        diff = "\n".join(difflib.Differ().compare(self.column_names_, column_names))
        message = (
            f"Columns of dataframes passed to fit() and transform() differ:\n{diff}"
        )
        raise ValueError(message)
