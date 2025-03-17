import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

import skrub


def test_parallel_coord():
    X_a, y_a = make_classification(n_samples=20, n_features=4, n_informative=2)
    c0 = skrub.choose_from({"a": 0, "b": 1}, name="c0")
    c1 = skrub.choose_from([0, 1], name="c1")

    c2 = skrub.choose_int(1, 100, log=True, name="c2")
    c3 = skrub.choose_float(0.0, 1.0, name="c3")
    c4 = skrub.choose_from({"aone": 101, "two": 102}, name="c4")
    c5 = skrub.choose_bool(name="c5")
    c6 = skrub.choose_from([2, 3, 4, 5], name="c6").match({2: c2, 3: c3, 4: c4, 5: c5})

    X = skrub.as_expr([skrub.X(), c0, c1, c6])[0]
    pred = X.skb.apply(DummyClassifier(), y=skrub.y())
    search = pred.skb.get_randomized_search(random_state=0, n_iter=30).fit(
        {"X": X_a, "y": y_a}
    )
    fig = search.plot_results()
    pytest.importorskip("plotly")
    data = iter(fig.data[0]["dimensions"])
    dim = next(data)
    assert dim["label"] == "score"
    dim = next(data)
    assert dim["label"] == "c0"
    assert list(dim["ticktext"]) == ["a", "b"]
    assert list(dim["tickvals"]) == [0, 1]
    # TODO: check other dimensions
