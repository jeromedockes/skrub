"""
Using the Recipe
================

Skrub provides the |Recipe|, a convenient and interactive way to create
machine-learning models for tabular data.

The |Recipe| is a helper to build a scikit-learn |Pipeline| (a chain of data
processing steps) and its ranges of hyperparameters (the parameters used to
initialize scikit-learn estimators, such as the number of trees in a random
forest).

The |Recipe| brings 3 major benefits:

- The pipeline can be built step by step, while easily checking the output of
  data transformations we have added so far. This makes development faster and
  more interactive.

- When we add a transformation, we can easily specify which columns it should
  modify. Tabular data is heterogeneous and many processing steps only apply to
  some of the columns in our dataframe.

- When we add a transformer or a final predictor, we can specify a range for
  any of its parameters, rather than a single value. Thus the ranges of
  hyperparameters to consider for tuning are provided directly inside the
  corresponding estimator, instead of being kept in a separate dictionary as is
  usually done with scikit-learn. Moreover, these hyperparameters can be given
  human-readable names which are useful for inspecting hyperparameter search
  results.

The Recipe is not an estimator
------------------------------

The recipe is a tool for configuring a scikit-learn estimator. It is not an
estimator itself (it does not have ``fit`` or ``predict``) methods. Once we
have created a Recipe, we must call one of its methods such as
``get_pipeline``, ``get_grid_search``, …, to obtain the corresponding
scikit-learn estimator.

.. image:: ../_static/recipe-graph.svg

For example:

- ``get_pipeline`` returns a scikit-learn |Pipeline| which applies all the
  steps we have configured without any hyperparameter tuning.
- ``get_grid_search`` returns a scikit-learn |GridSearchCV| which uses the same
  pipeline but runs a nested cross-validation loop to evaluate different
  combinations of hyperparameters from the ranges we have specified and select
  the best configuration.

.. |Pipeline| replace::
    :class:`~sklearn.pipeline.Pipeline`

.. |GridSearchCV| replace::
    :class:`~sklearn.model_selection.GridSearchCV`

.. |Recipe| replace::
     :class:`~skrub.Recipe`

.. |TableReport| replace::
     :class:`~skrub.TableReport`

.. |DatetimeEncoder| replace::
     :class:`~skrub.DatetimeEncoder`


"""


# %%
# Getting a preview of the transformed data
# -----------------------------------------
#
# The |Recipe| can be initialized with the full dataset (including the
# prediction target, ``y``). We can tell it which columns constitute the target
# and should be kept separate from the rest.

# %%
from skrub import Recipe, datasets

dataset = datasets.fetch_employee_salaries()
df = dataset.X
df["salary"] = dataset.y

recipe = Recipe(df, y_cols="salary", n_jobs=8)
recipe

# %%
# Our recipe does not contain any transformations yet, except for
# the built-in one which separates the target columns from the predictive
# features.

# %%
# We can use ``sample()`` to get a sample of the data transformed
# by the steps we have added to the recipe so far. The recipe draws a
# random sample from the dataframe we used to initialize it, and applies the
# transformations we have specified.
#
# At this point we have not added any transformations yet, so we just get a
# sample from the original dataframe.

# %%
recipe.sample()

# %%
# If instead of a random sample we want to see the transformation of the first
# few rows in their original order, we can use ``head()`` instead of ``sample()``.
#
# We can ask for a |TableReport| of the transformed sample, to inspect it
# more easily:

# %%
recipe.get_report()

# %%
# Adding transformations to specific columns
# ------------------------------------------
#
# We can use the report above to explore the dataset and plan the
# transformations we need to apply to the different columns.
# (Note that in the "Distributions" tab, we can select columns and construct a
# list of column names that we can copy-paste to save some typing.)
#
# We notice that the ``date_first_hired`` column contains dates such as
# ``'01/21/2001'`` but has the dtype ``ObjectDType`` -- which is the pandas
# representation of strings. Our dates are represented at strings, which we
# need to parse to transform them into proper datetime objects from which we
# can then extract information.
#
# Without Skrub, we might do that on the whole dataframe using something like:

# %%

# df["date_first_hired"] = pd.to_datetime(df["date_first_hired"])

# %%
# However, we would then need to remember to apply the same transformation when
# we receive new data for which we are asked to make a prediction. And we would
# also need to store the detected datetime format to be sure we apply the same
# transformation. Skrub helps us integrate that data-wrangling operation into
# our machine-learning model.
#
# To do so, we now add the first step to our recipe.

# %%
from skrub import ToDatetime
from skrub import selectors as s

recipe = recipe.add(ToDatetime(), cols="date_first_hired")
recipe

# %%
# To add a transformation, we call ``recipe.add()`` with the transformer to
# apply and optionally ``cols``, the set of columns it should modify.
#
# The columns can be a single column name (as shown above), or a list of column names.
# Skrub also provides more powerful column selectors as we will see below.
#
# Note that we wrote ``recipe = recipe.add(...)``. This is necessary because
# ``add()`` does not modify the recipe in-place. Instead, it returns a new
# |Recipe| object which contains the added step. To keep working with this new,
# augmented recipe, we bind the variable ``recipe`` to the new object.
#
# We can check what the data looks like with the added transformation.
# In the **"Show: ..."** dropdown, we can filter which columns we want to see in
# the report. The default is to show all of them, but if we select "Modified by
# last step", we see only those that were created or modified by the
# transformation we just added. In our case, this is the ``"date_first_hired"``
# column. We can see that it has a new dtype: it is now a true datetime column.

# %%
recipe.get_report()

# %%
# More flexible column selection
# ------------------------------
#
# Now that we have converted ``"date_first_hired"`` to datetimes, we can extract
# temporal features such as the year and month from it with Skrub's
# |DatetimeEncoder|.
#
# We could again specify ``cols="date_first_hired"`` but we will take this
# opportunity to introduce more flexible ways of selecting columns.
#
# The ``skrub.selectors`` module provides objects that we can use instead of
# an explicit list of column names, and which allow us to select columns based
# on their type, patterns in their name, etc. If you have used Polars or Ibis
# selectors before, they work in the same way.
#
# For example, ``skrub.selectors.any_date()`` selects all columns that have a
# datetime or date dtype.

# %%
from skrub import DatetimeEncoder

recipe = recipe.add(DatetimeEncoder(), cols=s.any_date())
recipe

# %%
# If we look at the preview, we see that ``any_date()`` has selected the
# ``"date_first_hired"`` column and that the |DatetimeEncoder| has extracted
# the year, month, day, and total number of seconds since the Unix epoch (start
# of 1970). You can easily check this with "Show: Modified by last step".

# %%
recipe.get_report()

# %%
# Selectors can be combined with the same operators as Python sets. For
# example, ``s.numeric() - "ID"`` would select all numeric columns except "ID",
# or ``s.categorical() | s.string()`` would get all columns that have either a
# Categorical or a string dtype.
#
# Here we use ``s.string() & s.cardinality_below(30)`` to select columns that
# contain strings *and* have a cardinality (number of unique values) strictly
# below 30. Those string columns with few unique values probably represent
# categories, so we will convert them to an actual ``Categorical`` dtype so
# that the rest of the pipeline can recognize and process them as such.
# High-cardinality string columns will be dealt with separately.


# %%
from skrub import ToCategorical

recipe = recipe.add(ToCategorical(), cols=s.string() & s.cardinality_below(30))
recipe

# %%
# Specifying alternative transformers and a hyperparameter grid
# -------------------------------------------------------------

# %%
from sklearn.preprocessing import TargetEncoder

from skrub import MinHashEncoder, choose_from

recipe = recipe.add(
    choose_from(
        {"target": TargetEncoder(), "minhash": MinHashEncoder()}, name="encoder"
    ),
    cols=s.string(),
)
recipe

# %%
recipe.get_report()

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

from skrub import choose_float

recipe = recipe.add(
    HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        learning_rate=choose_float(0.001, 1.0, log=True, name="learning rate"),
    )
)
recipe


# %%
# Getting the model & inspecting search results
# ---------------------------------------------

# %%
# randomized_search = recipe.get_randomized_search(n_iter=10, cv=3, verbose=1)
# randomized_search.fit(recipe.get_x_train(), recipe.get_y_train())

# recipe.get_cv_results_table(randomized_search)

# # %%
# recipe.plot_parallel_coord(randomized_search)
