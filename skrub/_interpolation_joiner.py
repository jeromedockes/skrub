import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.utils._tags import _safe_tags

from skrub import _utils
from skrub._table_vectorizer import TableVectorizer


class InterpolationJoiner(TransformerMixin, BaseEstimator):
    """Join with a table augmented by machine-learning predictions.

    This is similar to a usual equi-join, but instead of looking for actual
    rows in the right table that satisfy the join condition, we estimate what
    those rows would contain if they existed in the table.

    Suppose we want to join a table ``buildings(latitude, longitude, n_stories)``
    with a table ``annual_avg_temp(latitude, longitude, avg_temp)``. Our annual
    average temperature table may not contain data for the exact latitude and
    longitude of our buildings. However, we can interpolate what we need from
    the data points it does contain. Using ``annual_avg_temp``, we train a
    model to predict the temperature, given the latitude and longitude. Then,
    we use this model to estimate the values we want to add to our
    ``buildings`` table. In a way we are joining ``buildings`` to a virtual
    table, in which rows for any (latitude, longitude) location are inferred,
    rather than retrieved, when requested. This is done with::

        InterpolationJoiner(
            annual_avg_temp, on=["latitude", "longitude"]
        ).fit_transform(buildings)

    Parameters
    ----------
    aux_table : DataFrame
        The (auxiliary) table to be joined to the `main_table` (which is the
        argument of ``transform``). ``aux_table`` is used to train a model that
        takes as inputs the contents of the columns listed in ``aux_key``, and
        predicts the contents of the other columns. In the example above, we
        want our transformer to add temperature data to the table it is
        operating on. Therefore, ``aux_table`` is the ``annual_avg_temp``
        table.

    main_key : list of str, or str
        The columns in the main table used for joining. The main table is the
        argument of ``transform``, to which we add information inferred using
        ``aux_table``. The column names listed in ``main_key`` will provide the
        inputs (features) of the interpolators at prediction (joining) time. In
        the example above, ``main_key`` is ``["latitude", "longitude"]``, which
        refer to columns in the ``buildings`` table. When joining on a single
        column, we can pass its name rather than a list: ``"latitude"`` is
        equivalent to ``["latitude"]``.

    aux_key : list of str, or str
        The columns in ``aux_table`` used for joining. Their number and types
        must match those of the ``main_key`` columns in the main table. These
        columns provide the features for the estimators to be fitted. As for
        ``main_key``, it is possible to pass a string when using a single
        column.

    key : list of str, or str
        Column names to use for both `main_key` and `aux_key`, when they are
        the same. Provide either `key` (only) or both `main_key` and `aux_key`.

    suffix : str
        Suffix to append to the ``aux_table``'s column names. You can use it
        to avoid duplicate column names in the join.

    regressor : scikit-learn regressor or None
        Model used to predict the numerical columns of ``aux_table``. If
        ``None``, a ``HistGradientBoostingRegressor`` with default parameters
        is used.

    classifier : scikit-learn classifier or None
        Model used to predict the categorical (string) columns of
        ``aux_table``. If ``None``, a ``HistGradientBoostingRegressor`` with
        default parameters is used.

    vectorizer : scikit-learn transformer that can operate on a DataFrame or None
        Used to transform the feature columns before passing them to the
        scikit-learn estimators. This is useful if we are joining on columns
        that cannot be used directly, such as timestamps or strings
        representing high-cardinality categories. If ``None``, a
        ``TableVectorizer`` is used.

    n_jobs : int
        Number of joblib workers to use Depending on the estimators used and
        the contents of ``aux_table``, several estimators may need to be
        fitted -- for example one for continuous outputs (regressor) and one
        for categorical outputs (classifier), or one for each column when the
        provided estimators do not support multi-output tasks. Fitting and
        querying these estimators can be done in parallel.

    Attributes
    ----------
    vectorizer_ : scikit-learn transformer
        The transformer used to vectorize the feature columns.

    estimators_ : list of dicts
        The estimators used to infer values to be joined. Each entry in this
        list is a dictionary with keys ``"estimator"`` (the fitted estimator)
        and ``"columns"`` (the list of columns in ``aux_table`` that it is
        trained to predict).

    See Also
    --------
    Joiner :
        Works in a similar way but instead of inferring values, picks the
        closest row from the auxiliary table.

    Examples
    --------
    >>> buildings
       latitude  longitude  n_stories
    0       1.0        1.0          3
    1       2.0        2.0          7

    >>> annual_avg_temp
       latitude  longitude  avg_temp
    0       1.2        0.8      10.0
    1       0.9        1.1      11.0
    2       1.9        1.8      15.0
    3       1.7        1.8      16.0
    4       5.0        5.0      20.0

    >>> InterpolationJoiner(
    ...     annual_avg_temp,
    ...     key=["latitude", "longitude"],
    ...     regressor=KNeighborsRegressor(2),
    ... ).fit_transform(buildings)
       latitude  longitude  n_stories  avg_temp
    0       1.0        1.0          3      10.5
    1       2.0        2.0          7      15.5
    """

    def __init__(
        self,
        aux_table,
        *,
        main_key=None,
        aux_key=None,
        key=None,
        suffix="",
        regressor=None,
        classifier=None,
        vectorizer=None,
        n_jobs=1,
    ):
        self.aux_table = aux_table
        self.main_key = main_key
        self.aux_key = aux_key
        self.key = key
        self.suffix = suffix
        self.regressor = regressor
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.n_jobs = n_jobs

    def fit(self, X=None, y=None):
        """Fit estimators to the `aux_table` provided during initialization.

        `X` and `y` are for scikit-learn compatibility and they are
        ignored.

        Parameters
        ----------
        X : array-like
            Ignored.

        y : array-like
            Ignored.

        Returns
        -------
        self : InterpolationJoiner
            Returns self.
        """
        del X, y
        self._check_inputs()
        key_values = self.vectorizer_.fit_transform(self.aux_table[self._aux_key])
        estimators = self._get_estimator_assignments()
        self.estimators_ = joblib.Parallel(self.n_jobs)(
            joblib.delayed(_fit)(
                key_values,
                self.aux_table[assignment["columns"]],
                assignment["estimator"],
            )
            for assignment in estimators
        )
        return self

    def _check_inputs(self):
        if self.vectorizer is None:
            self.vectorizer_ = TableVectorizer()
        else:
            self.vectorizer_ = clone(self.vectorizer)

        if self.classifier is None:
            self.classifier_ = HistGradientBoostingClassifier()
        else:
            self.classifier_ = clone(self.classifier)

        if self.regressor is None:
            self.regressor_ = HistGradientBoostingRegressor()
        else:
            self.regressor_ = clone(self.regressor)

        self._check_key()

    def _check_key(self):
        """Find the correct main and auxiliary keys (matching column names).

        They can be provided either as ``key`` when the names are the same in
        both tables, or as ``main_key`` and ``aux_key`` when they differ. This
        function checks that only one of those options is used and sets
        ``self._main_key`` and ``self._aux_key`` which will be used for
        joining.
        """
        if self.key is not None:
            if self.aux_key is not None or self.main_key is not None:
                raise ValueError(
                    "Can only pass argument 'key' OR 'main_key' and "
                    "'aux_key', not a combination of both."
                )
            main_key, aux_key = self.key, self.key
        else:
            if self.aux_key is None or self.main_key is None:
                raise ValueError(
                    "Must pass EITHER 'key', OR ('main_key' AND 'aux_key')."
                )
            main_key, aux_key = self.main_key, self.aux_key
        self._main_key = _utils.atleast_1d_or_none(main_key)
        self._aux_key = _utils.atleast_1d_or_none(aux_key)

    def transform(self, X):
        """Transform a table by joining inferred values to it.

        The values of the `main_key` columns in `X` (the main table) are used
        to predict likely values for the contents of a matching row in
        `self.aux_table` (the auxiliary table).

        Parameters
        ----------
        X : DataFrame
            The (main) table to transform.

        Returns
        -------
        join : DataFrame
            The result of the join between `X` and inferred rows from
            ``self.aux_table``.
        """
        main_table = X
        key_values = self.vectorizer_.transform(main_table[self._main_key])
        interpolated_parts = joblib.Parallel(self.n_jobs)(
            joblib.delayed(_predict)(
                key_values, assignment["columns"], assignment["estimator"]
            )
            for assignment in self.estimators_
        )
        interpolated_parts = _add_column_name_suffix(interpolated_parts, self.suffix)
        for part in interpolated_parts:
            part.index = main_table.index
        return pd.concat([main_table] + interpolated_parts, axis=1)

    def _get_estimator_assignments(self):
        """Identify column groups to be predicted together and assign them an estimator.

        In many cases, a single estimator cannot handle all the target columns.
        This function groups columns that can be handled together and returns a
        list of dictionaries, each with keys "columns" and "estimator".

        Regression and classification targets are always handled separately.

        Any column with missing values is handled separately from the rest.
        This is due to the fact that missing values in the columns we are
        trying to predict have to be dropped, and the corresponding rows may
        have valid values in the other columns.

        When the estimator does not handle multi-output, an estimator is fitted
        separately to each column.
        """
        aux_table = self.aux_table.drop(self._aux_key, axis=1)
        assignments = []
        regression_table = aux_table.select_dtypes("number")
        assignments.extend(
            _get_assignments_for_estimator(regression_table, self.regressor_)
        )
        classification_table = aux_table.select_dtypes(["object", "string", "category"])
        assignments.extend(
            _get_assignments_for_estimator(classification_table, self.classifier_)
        )
        return assignments


def _get_assignments_for_estimator(table, estimator):
    """Get the groups of columns assigned to a single estimator.

    (which is either the regressor or the classifier)."""

    # If the complete set of columns that have to be predicted with this
    # estimator is empty (eg the estimator is the regressor and there are no
    # numerical columns), return an empty list -- no columns are assigned to
    # that estimator.
    if table.empty:
        return []
    if not _handles_multioutput(estimator):
        return [{"columns": [col], "estimator": estimator} for col in table.columns]
    columns_with_nulls = table.columns[table.isnull().any()]
    assignments = [
        {"columns": [col], "estimator": estimator} for col in columns_with_nulls
    ]
    columns_without_nulls = list(set(table.columns).difference(columns_with_nulls))
    if columns_without_nulls:
        assignments.append({"columns": columns_without_nulls, "estimator": estimator})
    return assignments


def _handles_multioutput(estimator):
    return _safe_tags(estimator).get("multioutput", False)


def _fit(key_values, target_table, estimator):
    estimator = clone(estimator)
    kept_rows = target_table.notnull().all(axis=1).to_numpy()
    key_values = key_values[kept_rows]
    Y = target_table.to_numpy()[kept_rows]

    # Estimators that expect a single output issue a DataConversionWarning if
    # passing a column vector rather than a 1-D array
    if len(target_table.columns) == 1:
        Y = Y.ravel()

    estimator.fit(key_values, Y)
    return {"columns": target_table.columns, "estimator": estimator}


def _predict(key_values, columns, estimator):
    Y_values = estimator.predict(key_values)
    return pd.DataFrame(data=Y_values, columns=columns)


def _add_column_name_suffix(dataframes, suffix):
    if suffix == "":
        return dataframes
    renamed = []
    for df in dataframes:
        renamed.append(df.rename(columns={c: f"{c}{suffix}" for c in df.columns}))
    return renamed
