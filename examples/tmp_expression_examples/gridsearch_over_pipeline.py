# rewrite https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_pipeline_display.html#displaying-a-grid-search-over-a-pipeline-with-a-classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import skrub
from skrub import selectors as s

X, y = skrub.X(), skrub.y()

num = (
    X.skb.select(s.numeric())
    .skb.apply(SimpleImputer(missing_values=float("nan"), strategy="mean"))
    .skb.apply(StandardScaler())
)

cat = (
    X.skb.drop(s.numeric())
    .skb.apply(
        SimpleImputer(fill_value="missing", missing_values=None, strategy="constant")
    )
    .skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
)

feat = num.skb.concat_horizontal([cat])

rf = RandomForestClassifier(
    n_estimators=skrub.choose_from([200, 500]),
    max_features=skrub.choose_from(["sqrt", "log2"]),
    max_depth=skrub.choose_int(4, 9),
    criterion=skrub.choose_from(["gini", "entropy"]),
)
pred = feat.skb.apply(rf, y=y)

search = pred.skb.get_grid_search()

# %%
# try the pipeline:
import polars as pl

X = pl.DataFrame({"a": [1.5, 10, None], "b": ["a", None, "b"]})
y = [1, 1, 0]

pred.skb.full_report(environment={"X": X, "y": y}).open()
