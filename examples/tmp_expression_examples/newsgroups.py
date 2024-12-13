import polars as pl
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import skrub
from skrub import selectors as s

categories = ["sci.med", "sci.space"]
X_train, y_train = fetch_20newsgroups(
    random_state=1,
    subset="train",
    categories=categories,
    remove=("footers", "quotes"),
    return_X_y=True,
)
X_test, y_test = fetch_20newsgroups(
    random_state=1,
    subset="test",
    categories=categories,
    remove=("footers", "quotes"),
    return_X_y=True,
)

X, y = skrub.X(X_train[:100]), skrub.y(y_train[:100])

# %%


@skrub.deferred
def extract_subject_body(posts):
    subjects, bodies = [], []
    for i, text in enumerate(posts):
        headers, _, body = text.partition("\n\n")
        prefix = "Subject:"
        sub = ""
        for line in headers.splitlines():
            if line.startswith(prefix):
                sub = line.removeprefix(prefix)
                break
        subjects.append(sub.strip())
        bodies.append(body.strip())

    return pl.DataFrame({"subject": subjects, "body": bodies})


@skrub.deferred
def add_text_stats(df):
    return df.with_columns(
        pl.all().str.len_bytes().name.suffix("__length"),
        pl.all().str.count_matches(".", literal=True).name.suffix("__num_sentences"),
    )


text_encoder = skrub.TextEncoder(
    "sentence-transformers/paraphrase-albert-small-v2",
    device="cpu",
)

# %%

X = extract_subject_body(X)
X = add_text_stats(X)
X = X.skb.apply(text_encoder, cols=s.string())
pred = X.skb.apply(LinearSVC(dual=False), y=y)

pred.skb.full_report().open()

# %%

estimator = pred.skb.get_estimator()
estimator.fit({"X": X_train, "y": y_train})

y_pred = estimator.predict({"X": X_test})
print("Classification report:\n\n{}".format(classification_report(y_test, y_pred)))
