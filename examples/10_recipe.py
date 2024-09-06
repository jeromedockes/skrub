"""
Using the Recipe
================

Skrub provides the Recipe, a convenient and interactive way to create simple or
complex machine-learning models for tabular data.

The Recipe is a helper to build a scikit-learn ``Pipeline`` and its grid of
hyperparameters.

It brings 3 major benefits:

- The pipeline can be built step by step, while easily checking the output of
  transformations we have added so far. This makes development faster and more
  interactive.

- When we add a transformation, we can easily specify on which columns it
  should be applied. Tabular data is heterogeneous and many processing steps
  only apply to some of our dataframe's columns.

- When we add a transformer or a final predictor, we can specify a range for
  any of its parameters, rather than a single value. Thus the ranges of
  hyperparameters to consider for tuning are provided directly inside the
  corresponding estimator, instead of being kept in a separate dictionary as is
  usually done with scikit-learn. Moreover, these hyperparameters can be given
  human-readable names which are useful for inspecting hyperparameter search
  results.

The Recipe is not an estimator
------------------------------

The recipe is a tool for configuring a scikit-learn estimator, it is not an
estimator itself (it does not have `fit` or `predict`) methods. Once we are
happy with our Recipe, we must call one of its methods such as
``get_pipeline``, ``get_grid_search``, …, to obtain the corresponding
scikit-learn estimator.

.. image:: ../_static/recipe-graph.svg

"""


# %%
# Getting a preview of the transformed data
# -----------------------------------------
#
# The ``Recipe`` can be initialized with the full dataset (including the
# target, ``y``) and we can tell it which columns are the target and should be
# kept separate from the rest.

# %%
from skrub import Recipe, datasets

dataset = datasets.fetch_employee_salaries()
df = dataset.X
df["salary"] = dataset.y

# %%
# At any point, we can use ``sample()`` to get a sample of the data transformed
# by the steps we have added to the recipe so far. The ``Recipe`` draws a
# random sample from the dataframe we used to initialize it, and applies the
# transformations we have specified.
#
# At this point we have not added any transformations yet, so we just get a
# sample from the original dataframe.

# %%
recipe = Recipe(df, y_cols="salary", n_jobs=8)
recipe.sample()

# %%
# If instead of a random sample we want to see the transformation of the first
# few rows in their original order, we can use ``head()`` instead of ``sample()``.
#
# We can ask for a ``TableReport`` of the transformed sample, to inspect it
# more easily:

# %%
recipe.get_report()

# %%
# Adding transformations to specific columns
# ------------------------------------------
#
# We can use the report above to explore the dataset and plan the
# transformations we need to apply to the different columns.
# In the "Column summaries" tab,

# %%
from skrub import (
    DatetimeEncoder,
    MinHashEncoder,
    ToCategorical,
    ToDatetime,
)
from skrub import selectors as s

recipe = (
    recipe.add(ToDatetime(), cols="date_first_hired")
    .add(DatetimeEncoder(), cols=s.any_date())
    .add(ToCategorical(), cols=s.string() & s.cardinality_below(30))
)
recipe

# %%
# Specifying alternative transformers and a hyperparameter grid
# -------------------------------------------------------------

# %%
from sklearn.preprocessing import TargetEncoder

from skrub import choose_from

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
randomized_search = recipe.get_randomized_search(n_iter=10, cv=3, verbose=1)
randomized_search.fit(recipe.get_x_train(), recipe.get_y_train())

# recipe.get_cv_results_table(randomized_search)

# # %%
# recipe.plot_parallel_coord(randomized_search)
