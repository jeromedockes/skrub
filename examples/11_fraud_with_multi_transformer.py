from skrub import datasets

# the actual data
dataset = datasets.fetch_credit_fraud()
baskets_df, products_df = dataset.baskets, dataset.products

# to simulate a situation where we have to construct the target: we separate
# "fraud_flag" (the target) from the baskets dataframe and shuffle it. Thus to
# get the target we will need to perform a join to re-align the fraud flags
# with the basket IDs.

fraud_flag_df = baskets_df[["ID", "fraud_flag"]]
fraud_flag_df = fraud_flag_df.sample(n=fraud_flag_df.shape[0])
baskets_df = baskets_df.drop("fraud_flag", axis="columns")

# %%
baskets_df

# %%
products_df

# %%
fraud_flag_df

# %%
import skrub as skb

baskets = skb.load("baskets", baskets_df).mark_as_x()
baskets

# %%
baskets.get_value()

# %%
baskets.get_report()

# %%
print(baskets.describe_steps())

# %%
baskets.draw_graph()

# %%

# the construction of the target -- we join it to the baskets to put the flags
# in the correct order then extract the flags column.

fraud_flag_raw = skb.load("fraud_flag", fraud_flag_df)
fraud_flag = (
    baskets.join(fraud_flag_raw, left_on="ID", right_on="ID")
    .select("fraud_flag")
    .mark_as_y()
)
fraud_flag.get_report()

# %%
products = skb.load("products", products_df)

# Add a 'total price' column to the products

total_price = products.select("Nbr_of_prod_purchas") * products.select("cash_price")
products = products.assign(total_price=total_price)

products.get_report()

# %%
from skrub import selectors as s

# Encode the string columns in the products: apply min-hash then aggregate by
# computing the minimum of each column.

product_strings = (
    products.select("basket_ID" | s.string())
    .apply(skb.MinHashEncoder(n_components=3), cols=s.all() - "basket_ID")
    .agg(by="basket_ID", func="min")
)
product_strings.get_report()


# %%

# Encode the numeric columns in the products: aggregate by computing the sum

product_numbers = products.select("basket_ID" | s.numeric()).agg(
    by="basket_ID", func="sum"
)
product_numbers.get_report()

# %%
# Now we join both on the baskets

encoded_baskets = (
    baskets.join(product_strings, "ID", "basket_ID")
    .join(product_numbers, "ID", "basket_ID")
    .drop("ID")
)
encoded_baskets.get_report()

# %%
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

learning_rate = skb.choose_from([0.1, 0.01], name="learning rate")
hgb = HistGradientBoostingClassifier(learning_rate=learning_rate)

dummy = DummyClassifier()

classifier = skb.choose_from({"hgb": hgb, "dummy": dummy}, name="classifier")

predicted_fraud = encoded_baskets.apply(classifier, y=fraud_flag)
predicted_fraud.get_report()

# %%
predicted_fraud.get_value()

# %%
predicted_fraud.draw_graph()

# %%
# In the graph above the blue and orange node denote X and y, which are
# evaluated before entering the cross-validation loop and define the cv splits.
#
# The output of each node is cached so it is only computed once. After each
# step the cache entries that will not be needed anymore are pruned.

# %%
print(predicted_fraud.describe_steps())

# %%
estimator = predicted_fraud.get_estimator()
tables = predicted_fraud.get_data()
tables

# %%
estimator.fit_tables(tables)
estimator.predict_tables(tables)

# %%

# get_randomized_search() also works
search = predicted_fraud.get_grid_search(cv=3, verbose=1, scoring="roc_auc", n_jobs=3)
search.fit_tables(tables)

print(search.get_cv_results_table())

# %%
search.plot_parallel_coord()

# %%
cv_results = skb.cross_validate(
    estimator,
    tables,
    cv=3,
    scoring="neg_log_loss",
    verbose=1,
    n_jobs=3,
)
# or cross_validate_tables
print(cv_results)

# %%

# just for fun, build an ensemble model

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

hgb_predictions = encoded_baskets.apply(HistGradientBoostingClassifier(), y=fraud_flag)
rf_predictions = encoded_baskets.apply(RandomForestClassifier(), y=fraud_flag)
all_predictions = hgb_predictions.cat(rf_predictions)
stacking_predictions = all_predictions.apply(LogisticRegression(), y=fraud_flag)

skb.cross_validate(
    stacking_predictions.get_estimator(),
    tables,
    cv=3,
    scoring="neg_log_loss",
    verbose=1,
    n_jobs=3,
)
