# https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py

from sklearn.datasets import fetch_openml

import skrub
from skrub import selectors as s

df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
MAX = df["count"].max()

# %%
bikes = skrub.var("bikes", df)
X = bikes.drop("count", axis="columns").skb.mark_as_x()
y = (bikes["count"] / MAX).skb.mark_as_y()

# %%
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
)


def periodic_spline_transformer(period, n_splines, degree=3):
    n_knots = n_splines + 1
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


hour_workday = (
    X[["hour", "workingday"]]
    .skb.apply(periodic_spline_transformer(24, 8), cols="hour")
    .assign(workingday=X["workingday"] == "True")
    .skb.apply(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
)

# %%
X = (
    X.skb.apply(periodic_spline_transformer(12, 6), cols="month")
    .skb.apply(periodic_spline_transformer(7, 3), cols="weekday")
    .skb.apply(periodic_spline_transformer(24, 12), cols="hour")
    .skb.apply(MinMaxScaler(), cols=s.numeric() - ["month", "hour", "weekday"])
    .skb.apply(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols=~s.numeric()
    )
)

# %%
X_interactions = X.skb.concat_horizontal([hour_workday])

# %%
from sklearn.kernel_approximation import Nystroem

X_nystroem = X.skb.apply(
    Nystroem(kernel="poly", degree=2, n_components=300, random_state=0),
)

# %%
X_nystroem.skb.get_report().open()

# %%
from sklearn.linear_model import RidgeCV

pred = X_nystroem.skb.apply(RidgeCV(alphas=np.logspace(-6, 6, 25)), y=y)
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
