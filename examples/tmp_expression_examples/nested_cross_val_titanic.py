from sklearn.datasets import fetch_openml

import skrub
from skrub import selectors as s

feat_df, target_df = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

feat = skrub.var("features", feat_df).skb.mark_as_x()
target = skrub.var("target", target_df).skb.mark_as_y()

feat

# %%
from sklearn.preprocessing import OneHotEncoder

feat = feat.skb.apply(
    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    cols=["sex", "embarked"],
).skb.apply(skrub.MinHashEncoder(n_components=10), cols=~s.numeric())

feat

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

pred = feat.skb.apply(
    HistGradientBoostingClassifier(
        learning_rate=skrub.choose_float(0.001, 0.1, log=True, name="lr")
    ),
    y=target,
)

pred

# %%
search = pred.skb.get_randomized_search(verbose=1)

# single param search
search.fit({"features": feat_df, "target": target_df})
print(search.get_cv_results_table())

# %%

# cross-validating the param search
cv = skrub.cross_validate(
    search, {"features": feat_df, "target": target_df}, verbose=1, n_jobs=5
)
print(cv)
