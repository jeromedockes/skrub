# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import skrub
from skrub import selectors as s

df = fetch_openml("titanic", version=1, as_frame=True).frame

data = skrub.var("data", df)
X = data[["embarked", "sex", "pclass", "age", "fare"]].skb.mark_as_x()
y = data["survived"].skb.mark_as_y()

num = (
    X.skb.select(s.numeric())
    .skb.apply(
        SimpleImputer(
            strategy=skrub.choose_from(["median", "mean"], name="imputer strategy")
        )
    )
    .skb.apply(StandardScaler())
)

cat = (
    X.skb.select(~s.numeric())
    .skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    .skb.apply(
        SelectPercentile(chi2, percentile=skrub.choose_int(10, 70, name="percentile")),
        y=y,
    )
)

X = num.skb.concat_horizontal([cat])

pred = X.skb.apply(
    LogisticRegression(C=skrub.choose_float(0.1, 100, log=True, name="C")), y=y
)
clf = pred.skb.get_randomized_search()


df_train, df_test = train_test_split(df)
clf.fit({"data": df_train})
print(clf.get_cv_results_table())

prediction = clf.predict({"data": df_test})
print("model score: %.3f" % roc_auc_score(df_test["survived"], prediction))
