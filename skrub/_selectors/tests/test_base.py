def test_repr():
    """
    >>> from skrub import _selectors as s
    >>> s.all()
    all()
    >>> s.all() - ["ID", "Name"]
    (all() - cols('ID', 'Name'))
    >>> "Name" & s.inv(["ID"]) | s.numeric()
    ((cols('Name') & (~cols('ID'))) | numeric())
    >>> ((s.cols('Name') & (~s.cols('ID'))) | s.numeric())
    ((cols('Name') & (~cols('ID'))) | numeric())
    >>> s.numeric() | s.glob("*_mm") - s.regex(r"^[ 0-9]+_mm$")
    (numeric() | (glob('*_mm') - regex('^[ 0-9]+_mm$')))
    >>> s.filter_names(lambda n: 'a' in n) ^ s.filter(lambda c: c[2] == 3)
    (filter_names(<function <lambda> at ...>) ^ filter(<function <lambda> at ...>))
    """
