import pytest

import skrub
from skrub import datasets


def test_output_dir(tmp_path):
    e = skrub.X()
    assert e.skb.full_report(open=False)["report_path"].is_relative_to(
        datasets.get_data_dir()
    )
    out = tmp_path / "report"
    assert (
        e.skb.full_report(open=False, output_dir=out)["report_path"]
        == out / "index.html"
    )
    with pytest.raises(FileExistsError, match=".*Set 'overwrite=True'"):
        e.skb.full_report(open=False, output_dir=out)

    assert (
        e.skb.full_report(open=False, output_dir=out, overwrite=True)["report_path"]
        == out / "index.html"
    )


def test_full_report():
    # smoke test for the full report
    # TODO we should have a private function that returns the JSON data so we
    #      can check the content before rendering with jinja
    # however that requires first settling on the content of the report etc.
    e = -(
        (skrub.var("a", 12345) + 1).skb.set_name("b").skb.set_description("this is b")
        / skrub.var("c", 1)
    )
    report = e.skb.full_report(open=False)
    assert report["error"] is None
    assert report["result"] == -12346.0
    assert "-12346.0" in (report["report_path"].parent / "node_4.html").read_text(
        "utf-8"
    )
    report = e.skb.full_report({"a": 12345, "c": 0}, open=False)
    assert isinstance(report["error"], (ZeroDivisionError, RuntimeError))
    assert report["result"] is None
    out = report["report_path"].parent
    text = (out / "node_1.html").read_text("utf-8")
    assert "12346" in text and "this is b" in text
    assert "ZeroDivisionError" in (out / "node_3.html").read_text("utf-8")
    assert "This step did not run" in (out / "node_4.html").read_text("utf-8")


def test_describe_param_grid():
    """
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.preprocessing import StandardScaler, RobustScaler

    >>> import skrub

    >>> X = skrub.X()
    >>> y = skrub.y()

    >>> imputed = X.skb.apply(skrub.optional(SimpleImputer, name="impute"))
    >>> dim_reduction = skrub.choose_from(
    ...     {
    ...         "PCA": PCA(),
    ...         "SelectKBest": SelectKBest(),
    ...     },
    ...     name="dim_reduction",
    ... )
    >>> selected = imputed.skb.apply(dim_reduction)
    >>> use_scaling = skrub.choose_bool(name="scaling")
    >>> scaling_kind = skrub.choose_from(["robust", "standard"], name="scaling_kind")
    >>> scaler = scaling_kind.match(
    ...     {
    ...         "robust": RobustScaler(
    ...             **skrub.choose_bool(name="robust_scaler__with_centering")
    ...         ),
    ...         "standard": StandardScaler(),
    ...     }
    ... )
    >>> scaled = selected.skb.apply(use_scaling.if_else(scaler, None))
    >>> classifier = skrub.choose_from(
    ...     {
    ...         "logreg": LogisticRegression(
    ...             **skrub.choose_float(0.001, 100, log=True, name="C")
    ...         ),
    ...         "rf": RandomForestClassifier(
    ...             n_estimators=skrub.choose_int(20, 400, name="N 🌴")
    ...         ),
    ...     },
    ...     name="classifier",
    ... )
    >>> pred = scaled.skb.apply(classifier, y=y)
    >>> print(pred.skb.describe_param_grid())
    - dim_reduction: ['PCA', 'SelectKBest']
      impute: ['true', 'false']
      classifier: 'logreg'
      C: choose_float(0.001, 100, log=True, name='C')
      scaling: True
      scaling_kind: 'robust'
      robust_scaler__with_centering: [True, False]
    - dim_reduction: ['PCA', 'SelectKBest']
      impute: ['true', 'false']
      classifier: 'logreg'
      C: choose_float(0.001, 100, log=True, name='C')
      scaling: True
      scaling_kind: 'standard'
    - dim_reduction: ['PCA', 'SelectKBest']
      impute: ['true', 'false']
      classifier: 'logreg'
      C: choose_float(0.001, 100, log=True, name='C')
      scaling: False
    - dim_reduction: ['PCA', 'SelectKBest']
      impute: ['true', 'false']
      classifier: 'rf'
      N 🌴: choose_int(20, 400, name='N 🌴')
      scaling: True
      scaling_kind: 'robust'
      robust_scaler__with_centering: [True, False]
    - dim_reduction: ['PCA', 'SelectKBest']
      impute: ['true', 'false']
      classifier: 'rf'
      N 🌴: choose_int(20, 400, name='N 🌴')
      scaling: True
      scaling_kind: 'standard'
    - dim_reduction: ['PCA', 'SelectKBest']
      impute: ['true', 'false']
      classifier: 'rf'
      N 🌴: choose_int(20, 400, name='N 🌴')
      scaling: False
    """
