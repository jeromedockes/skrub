import skrub
from skrub import datasets

data = datasets.fetch_credit_fraud()

# %%
baskets = skrub.var("baskets", data.baskets.drop(columns="fraud_flag")).skb.mark_as_x()
baskets

# %%
products = skrub.var("products", data.products)
products

# %%
fraud_flag = skrub.var(
    "fraud_flag",
    data.baskets[["ID", "fraud_flag"]]
    .set_axis(["basket_ID", "fraud_flag"], axis=1)
    .sample(frac=1.0),
)
fraud_flag

# %%
fraud_flag = baskets.merge(fraud_flag, left_on="ID", right_on="basket_ID")[
    "fraud_flag"
].skb.mark_as_y()
fraud_flag

# %%
total_price = products["Nbr_of_prod_purchas"] * products["cash_price"]
products = products.assign(total_price=total_price)

products[products["Nbr_of_prod_purchas"] > 1]

# %%
estimator_kind = skrub.choose_from(["rf", "hgb", "ridge"], name="estimator kind")
using_trees = estimator_kind.match({"rf": True, "hgb": True, "ridge": False})

min_hash = skrub.MinHashEncoder(
    n_components=skrub.choose_int(20, 100, log=True, name="# hash")
)
gap = skrub.GapEncoder(n_components=skrub.choose_int(5, 30, log=True, name="# topics"))

encoder = using_trees.match({True: min_hash, False: gap})

# %%
from skrub import selectors as s

product_strings = (
    products.skb.select("basket_ID" | s.string())
    .skb.apply(encoder, cols=s.all() - "basket_ID")
    .groupby(by="basket_ID")
    .agg(using_trees.match({True: "min", False: "mean"}))
    .reset_index()
)

# %%
baskets = baskets.merge(product_strings, left_on="ID", right_on="basket_ID").drop(
    "basket_ID", axis="columns"
)
baskets

# %%
product_numbers = (
    products.skb.select("basket_ID" | s.numeric())
    .groupby(by="basket_ID")
    .sum()
    .reset_index()
)

# %%
baskets = baskets.merge(product_numbers, left_on="ID", right_on="basket_ID").drop(
    ["ID", "basket_ID"], axis="columns"
)
baskets

# %%
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

classifier = estimator_kind.match(
    {
        "rf": RandomForestClassifier(
            n_estimators=skrub.choose_int(5, 200, log=True, name="# 🌲")
        ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=skrub.choose_float(0.001, 2.0, log=True, name="learning rate")
        ),
        "ridge": RidgeClassifier(alpha=skrub.choose_float(0.01, 1000.0, name="α")),
    }
)

prediction = baskets.skb.apply(classifier, y=fraud_flag)
prediction

# %%
prediction.skb.draw_graph()

# %%
print(prediction.skb.describe_steps())

# %%
print(prediction.skb.describe_param_grid())

# %%
prediction.skb.get_report().open()

# %%
prediction.skb.full_report()

# %%
search = prediction.skb.get_randomized_search(
    scoring="roc_auc", n_iter=32, n_jobs=16, cv=4, verbose=3
).fit(prediction.skb.get_data())

# %%
print(search.get_cv_results_table())

# %%
search.plot_parallel_coord().show()

# %%
cv_result = prediction.skb.cross_validate(scoring="roc_auc", n_jobs=8, verbose=3)
print(cv_result)
