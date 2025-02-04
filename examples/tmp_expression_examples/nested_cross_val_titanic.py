from sklearn.datasets import fetch_openml

import skrub
from skrub import selectors as s

features, target = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

X = skrub.X(features)
y = skrub.y(target)
X

# %%
from sklearn.preprocessing import OneHotEncoder

X = X.skb.apply(
    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    cols=["sex", "embarked"],
).skb.apply(skrub.MinHashEncoder(n_components=10), cols=~s.numeric())

X

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

pred = X.skb.apply(
    HistGradientBoostingClassifier(
        learning_rate=skrub.choose_float(0.001, 0.1, log=True, name="lr")
    ),
    y=y,
)

pred

# %%
search = pred.skb.get_randomized_search(fitted=True, verbose=1)

print(search.get_cv_results_table())

# %%

# cross-validating the param search
cv = skrub.cross_validate(search, {"X": features, "y": target}, verbose=1, n_jobs=5)
print(cv)
