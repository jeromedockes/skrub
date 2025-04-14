"""
Building complex tabular pipelines
==================================

Skrub provides an easy way to build complex, flexible machine-learning
pipelines. There are several problems that are not easily addressed with the
standard scikit-learn tools such as the ``Pipeline`` and ``ColumnTransformer``
and for which a skrub pipeline is a solution:

- **Multiple tables:** when there are several tables of different shapes (for example,
  "Customers", "Orders" and "Products" tables). Not suitable for scikit-learn
  estimators, expecting a single design matrix ``X`` and array of targets ``y``
  with one row per observation.

- **DataFrame wrangling:** performing typical dataframe operations
  such as projections, joins and aggregations leveraging the powerful APIs of
  ``pandas`` or ``polars``.

- **Iterative development:** building a pipeline step by step, while
  inspecting the intermediate results so that the feedback loop is short and
  errors are discovered early.

- **Hyperparameter tuning:** choices of estimators, hyperparameters,
  and even the architecture of the pipeline can be informed by validation scores.
  Specifying the grid of hyperparameters separately from the model (as in
  ``GridSearchCV``) is difficult for complex pipelines.

In this example, we show a pipeline to handle a dataset with 2 tables.
Despite being very simple, this pipeline would be difficult to implement,
validate and deploy correctly without skrub.
We show the script with minimal comments to motivate the tools that are
explained in detail in subsequent examples.
"""

# %%
# The credit fraud dataset
# ------------------------
#
# This dataset comes from an e-commerce website. We have a set of "baskets" (
# orders that have been placed with the website). The task is to detect which
# orders were fraudulent (the customer never made the payment).
#
# The ``baskets`` table contains a basket ID and a flag indicating if the order
# was fraudulent or not.

# %%
import skrub
import skrub.datasets

dataset = skrub.datasets.fetch_credit_fraud()
skrub.TableReport(dataset.baskets)

# %%
# Each basket contains one or more products. Each row in the ``products`` table
# corresponds to a type of product present in a basket. Products can
# be associated with the corresponding basket through the ``"basket_ID"``
# column.

# %%
skrub.TableReport(dataset.products)

# %%
# A data-processing challenge
# ----------------------------
#
# We want to fit a ``HistGradientBoostingClassifier`` to predict the fraud
# flag (or ``y``). We build a design matrix with one row per basket (and thus per
# fraud flag). Our ``baskets`` (or ``X``) table only contains IDs. We enrich it by
# adding features from the ``products`` table.
#
# The general structure of the pipeline looks like this:
#
# .. image:: ../../_static/credit_fraud_diagram.svg
#    :width: 300
#
#
# First, as the ``products`` table contains strings and categories (such as
# ``"SAMSUNG"``), we vectorize those entries to extract numeric
# features. This is easily done with skrub's ``TableVectorizer``. Then, as each
# basket can contain several products, all the product lines corresponding to a
# basket are aggregated into a single feature vector that can be
# attached to the basket.
#
# The difficulty is that the vectorized ``products`` should be aggregated before joining
# to ``baskets``, and, in order to compute a meaningful aggregation, must
# be vectorized *before* the aggregation. Thus, we have a ``TableVectorizer`` to
# fit on a table which does not (yet) have the same number of rows as the
# target ``y`` — something that the scikit-learn ``Pipeline``, with its
# single-input, linear structure, does not allow.
#
# We can fit it ourselves, outside of any pipeline with something like::
#
#     vectorizer = skrub.TableVectorizer()
#     vectorized_products = vectorizer.fit_transform(dataset.products)
#
# However, because it is dissociated from the main estimator which handles
# ``X`` (the baskets), we have to manage this transformer ourselves. We lose
# the scikit-learn machinery for grouping all transformation steps,
# storing fitted estimators, splitting the input data and cross-validation, and
# hyper-parameter tuning.
#
# Moreover, we might need some pandas code to perform the aggregation and join.
# Again, as this transformation is not in a scikit-learn estimator, it is error-prone.
# The difficulty is that we have to keep track of it ourselves to apply it later
# to unseen data, and we cannot tune any choices (like the choice of the
# aggregation function).
#
# Fortunately, skrub provides an alternative way to build
# more flexible pipelines.

# %%
# A solution with skrub
# ---------------------
#
# In a skrub pipeline, we do not provide an explicit list of
# transformation steps. Rather, we manipulate skrub objects representing
# intermediate results. The pipeline is built implicitly as we perform
# operations (such as applying operators or calling functions) on those
# objects.

# %%
# We start by creating skrub variables, which are the inputs to our pipeline.
# In our example, we create three variables: "products", "baskets", and "fraud flags":

# %%
products = skrub.var("products", dataset.products)
baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_X()
fraud_flags = skrub.var("fraud_flags", dataset.baskets["fraud_flag"]).skb.mark_as_y()

# %%
# They are given a name and an (optional) initial
# value, used to show previews of the pipeline's output, detect errors
# early, and provide data for cross-validation and hyperparameter search.
#
# We then build the pipeline by applying transformations to those inputs.
#
# Above, ``mark_as_X()`` and ``mark_as_y()`` indicate that the baskets and
# flags are respectively our design matrix and target variables, that
# should be split into training and testing sets for cross-validation. Here,
# they are direct inputs to the pipeline but any
# intermediate result could be marked as X or y.
#
# Because our pipeline expects dataframes for products, baskets and fraud
# flags, we manipulate those objects as we would manipulate dataframes.
# All attribute accesses are transparently forwarded to the actual input
# dataframes when we run the pipeline.
#
# For instance, we filter products to keep only those that match one of the
# baskets in the ``baskets`` table, and then add a column containing the total
# amount for each kind of product in a basket:

# %%
products = products[products["basket_ID"].isin(baskets["ID"])]
products = products.assign(
    total_price=products["Nbr_of_prod_purchas"] * products["cash_price"]
)
products

# %%
# We see previews of the output of intermediate results. For
# example, the added ``"total_price"`` column is in the output above.
# The "Show graph" dropdown at the top allows us to check the
# structure of the pipeline and all the steps it contains.
#
# With skrub, we do not need to specify a grid of hyperparameters separately
# from the pipeline. Instead, we replace a parameter's value with a skrub
# "choice" which indicates the range of values we consider during
# hyperparameter selection.
#
# Skrub choices can be nested arbitrarily. They are not restricted to
# parameters of a scikit-learn estimator, but can be anything: choosing
# between different estimators, arguments to function calls, whole sections of
# the pipeline etc.
#
# In-depth information about choices and hyperparameter/model selection is
# provided in the :ref:`Tuning Pipelines example <example_tuning_pipelines>`.
#
# We build a skrub ``TableVectorizer`` with different choices of:
# the type of encoder for high-cardinality categorical or string columns, and
# the number of components it uses.

# %%
n = skrub.choose_int(5, 15, log=True, name="n_components")
encoder = skrub.choose_from(
    {
        "MinHash": skrub.MinHashEncoder(n_components=n),
        "LSA": skrub.StringEncoder(n_components=n),
    },
    name="encoder",
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder)

# %%
# A transformer does not have to apply to the full dataframe; we can
# restrict it to some columns, using the ``cols`` or ``exclude_cols``
# parameters. In our example, we vectorize all columns except the ``"basket_ID"``.

# %%
vectorized_products = products.skb.apply(vectorizer, exclude_cols="basket_ID")

# %%
# Having access to the underlying dataframe's API, we can perform the
# data-wrangling we need. Those transformations are being implicitly added
# as steps in our machine-learning pipeline.

# %%
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
augmented_baskets = baskets.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
).drop(columns=["ID", "basket_ID"])

# %%
# We can actually ask for a full report of the pipeline and inspect the
# results at every step::
#
#     predictions.skb.full_report()
#
# This produces a folder on disk rather than displaying inline in a notebook so
# we do not run it here. But you can
# `see the output here <../../_static/credit_fraud_report/index.html>`_.
#
# Finally, we add a supervised estimator:

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
)
predictions = augmented_baskets.skb.apply(hgb, y=fraud_flags)
predictions

# %%
# And our pipeline is complete!
#
# From the choices we inserted at different locations in our pipeline, skrub
# can build a grid of hyperparameters and run the hyperparameter search for us,
# backed by scikit-learn's ``GridSearchCV`` or ``RandomizedSearchCV``.

# %%
print(predictions.skb.describe_param_grid())

# %%
search = predictions.skb.get_randomized_search(
    scoring="roc_auc", n_iter=8, n_jobs=4, random_state=0, fitted=True
)
search.results_

# %%
# We can also display a parallel coordinates plot of the results.
#
# In a parallel coordinates plot, each line corresponds to a combination
# of hyperparameter (choices) values, followed by the corresponding test
# scores, and training and scoring computation durations.
# Different columns show the hyperparameter values.
#
# By clicking and dragging the mouse on any column, we can restrict the
# set of lines we see. This allows quickly inspecting which hyperparameters are
# important, which values perform best, and potential trade-offs between the quality
# of predictions and computation time.

# %%
search.plot_results()

# %%
# It seems here that using the LSA as an encoder brings better test scores,
# but at the expense of training and scoring time.
#

# %%
# Conclusion
# ----------
#
# If you are curious to know more on how to
# build your own complex, multi-table pipelines with easy hyperparameter
# tuning, please see the next examples for an in-depth tutorial.
