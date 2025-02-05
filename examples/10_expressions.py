"""
Introducing skrub expressions
=============================

Skrub expressions are a way to build complex, flexible machine-learning
pipelines. They solve several problems that are not easily addressed with the
standard scikit-learn tools such as the ``Pipeline`` and ``ColumnTransformer``.

A machine-learning estimator may need to transform and extract information from
several tables of different shapes (for example, we may have "Customers",
"Orders" and "Products" tables). But scikit-learn estimators (including the
``Pipeline``) expect their input to be a single design matrix ``X`` and an
array of targets ``y`` in which each row corresponds to a sample. They do not
easily accommodate operations that change the number of rows such as aggregating
or filtering, or that combine several tables, such as joins.

The required transformations often involve a mix of scikit-learn estimators to
fit, and of operations on dataframes such as aggregations and joins. Often,
transformations should only be applied to some of the columns in the input
table. These requirements can be met using scikit-learn's
``FunctionTransformer``, ``Pipeline``, ``ColumnTransformer`` and
``FeatureUnion`` but this can become verbose and difficult to maintain.

Declaring all the steps in a pipeline before fitting it to see the result can
result in a slow development cycle, in which mistakes in the early steps of the
pipeline are only discovered later, when we fit the pipeline. We would like a
more interactive process where we immediately obtain previews of the
intermediate results (or errors).

A machine-learning pipeline involves many choices, such as which tables to use,
which features to construct and include, which estimators to fit, and the
estimators' hyperparameters. We want to provide a range of possible outcomes
for these choices, and use validation scores to select the best option for each
(hyperparameter tuning). Scikit-learn offers ``GridSearchCV``,
``RandomizedSearchCV`` and their halving counterparts to perform the
hyperparameter tuning. However, the grid of possible hyperparameters must be
provided separately from the pipeline itself. This can get cumbersome for
complex pipelines, especially when we want to tune not only simple
hyperparameters but also more structural aspects of the pipeline such as which
estimators to use.

Skrub can help us tackle these challenges. In this example, we show a pipeline
to handle a dataset with 2 tables. Despite being very simple, this pipeline
would be difficult to implement, validate and deploy correctly without skrub.
We leave out hyperparameter tuning, which is covered in the next example.
"""

# %%
# The credit fraud dataset
# ------------------------
#
# This dataset comes from an e-commerce website. We have a set of "baskets",
# orders that have been placed with the website. Some of those orders were
# fraudulent: the customer made a payment that was later declined by the credit
# card company. Our task is to detect which baskets correspond to a fraudulent
# transaction.
#
# The ``baskets`` table only contains a basket ID and the flag indicating if it
# was fraudulent or not.

# %%
import skrub
import skrub.datasets

# Use a richer representation of dataframes in the displayed output.
skrub.patch_display()

dataset = skrub.datasets.fetch_credit_fraud()
dataset.baskets

# %%
# Each basket contains one or more products. We have a ``products`` table
# detailing the actual content of each basket. Each row in the ``products``
# table corresponds to a type of product that was present in the basket
# (multiple units may have been bought, which is why there is a ``"Nbr"``
# column). Products can be associated with their basket through the
# ``"basket_ID"`` column.

# %%
dataset.products

# %%
# A data-processing challenge
# ----------------------------
#
# Our end-goal is to fit a supervised learner (a
# ``HistGradientBoostingClassifier``) to predict the fraud flag. To do this, we
# need to build a design matrix in which each row corresponds to a basket (and
# thus to a value in the ``fraud_flag`` column). At the moment, our ``baskets``
# table only contains IDs. We need to enrich it by adding features constructed
# from the actual contents of the baskets, that is, from the ``products``
# table.
#
# As the ``products`` table contains strings and categories (such as
# ``"SAMSUNG"``), we need to vectorize those entries to extract numeric
# features. This is easily done with skrub's ``TableVectorizer``. As each
# basket can contain several products, all the product lines corresponding to a
# basket then need to be aggregated, in order to produce a single feature
# vector that can be attached to the basket (associated with a fraud flag) and
# used to train our ``HistGradientBoostingClassifier``.
#
# Thus the general structure of the pipeline looks like this:
#
# .. image:: ../../_static/credit_fraud_diagram.svg
#    :width: 300
#
# We can see the difficulty: the products need to be aggregated before joining
# to ``baskets``, and in order to compute a meaningful aggregation, they must
# be vectorized _before_ the aggregation. So we have a ``TableVectorizer`` to
# fit on a table which does not (yet) have the same number of rows as the
# target ``y`` — something that the scikit-learn ``Pipeline``, with its
# single-input, linear structure, does not accommodate.
# We can fit it ourselves, outside of any pipeline with something like::
#
#     vectorizer = skrub.TableVectorizer()
#     vectorized_products = vectorizer.fit_transform(products)
#
# However, because it is dissociated from the main estimator which handles
# ``X`` (the baskets), we have to manage this transformer ourselves. We lose
# the usual scikit-learn machinery for grouping all transformation steps,
# storing fitted estimators, splitting the input data and cross-validation, and
# hyper-parameter tuning.
#
# Moreover, we later need some pandas code to perform the aggregation and join::
#
#     aggregated_products = (
#        vectorized_products.groupby("basket_ID").agg("mean").reset_index()
#     )
#     baskets = baskets.merge(
#         aggregated_products, left_on="ID", right_on="basket_ID"
#     ).drop(columns=["ID", "basket_ID"])
#
#
# Again, as this transformation is not in a scikit-learn estimator, we have to
# keep track of it ourselves so that we can later apply to unseen data, which
# is error-prone, and we cannot tune any choices (like the choice of the
# aggregation function).
#
# To cope with these difficulties, skrub provides a alternative way to build
# more flexible pipelines.
