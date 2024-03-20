from skrub import _dataframe as sbd
from skrub import _selectors as s


def test_repr():
    """
    >>> from skrub import _selectors as s
    >>> s.numeric() - s.boolean()
    (numeric() - boolean())
    >>> s.numeric() | s.glob("*_mm") - s.regex(r"^[ 0-9]+_mm$")
    (numeric() | (glob('*_mm') - regex('^[ 0-9]+_mm$')))
    >>> s.cardinality_below(30)
    cardinality_below(30)
    >>> s.string() | s.any_date() | s.categorical()
    ((string() | any_date()) | categorical())

    """


def test_glob(df_module):
    df = df_module.example_dataframe
    assert s.glob("*").expand(df) == s.all().expand(df)
    assert s.glob("xxx").expand(df) == []
    assert (s.glob("[Ii]nt-*") | s.glob("?loat-col")).expand(df) == [
        "int-col",
        "float-col",
    ]


def test_regex(df_module):
    df = df_module.example_dataframe
    assert (s.regex("int-.*") | s.regex("float-") | s.regex("date-$")).expand(df) == [
        "int-col",
        "float-col",
    ]


def test_dtype_selectors(df_module):
    df = df_module.example_dataframe
    cat_col = sbd.rename(sbd.to_categorical(sbd.col(df, "str-col")), "cat-col")
    df = sbd.make_dataframe_like(df, sbd.to_column_list(df) + [cat_col])
    assert s.numeric().expand(df) == ["int-col", "float-col"]
    assert (s.numeric() | s.boolean()).expand(df) == [
        "int-col",
        "float-col",
        "bool-col",
    ]
    assert s.string().expand(df) == ["str-col"]
    assert s.categorical().expand(df) == ["cat-col"]


def test_cardinality_below(df_module, monkeypatch):
    df = df_module.example_dataframe
    assert s.cardinality_below(3).expand(df) == ["bool-col"]
    assert s.cardinality_below(4).expand(df) == (s.all() - "date-col").expand(df)
    assert s.cardinality_below(5).expand(df) == s.all().expand(df)

    def bad_n_unique(c):
        raise ValueError()

    monkeypatch.setattr(sbd, "n_unique", bad_n_unique)
    assert s.cardinality_below(5).expand(df) == []
