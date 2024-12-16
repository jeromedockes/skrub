# https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py

from sklearn.datasets import fetch_openml

import skrub

df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
MAX = df["count"].max()

# %%
bikes = skrub.var("bikes", df)
X = bikes.drop("count", axis="columns").skb.mark_as_x()
y = (bikes["count"] / MAX).skb.mark_as_y()

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

gbrt = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)
pred = X.skb.apply(gbrt, y=y)
print(pred)

# %%
pred_orig_scale = pred * MAX
pred_orig_scale.skb.draw_graph().open()
print(pred_orig_scale)

# %%
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10000,
    test_size=1000,
)

# %%
res = pred.skb.cross_validate(
    cv=ts_cv, scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"]
)
print(f"rmse: {-res['test_neg_root_mean_squared_error'].mean():.3f}")
print(f"mae: {-res['test_neg_mean_absolute_error'].mean():.3f}")
