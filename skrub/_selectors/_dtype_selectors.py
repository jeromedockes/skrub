from .. import _dataframe as sbd
from ._base import Filter


class Numeric(Filter):
    def __init__(self):
        super().__init__(sbd.is_numeric)

    def __repr__(self):
        return "numeric()"


def numeric():
    return Numeric()


class AnyDate(Filter):
    def __init__(self):
        super().__init__(sbd.is_any_date)

    def __repr__(self):
        return "any_date()"


def any_date():
    return AnyDate()


class Categorical(Filter):
    def __init__(self):
        super().__init__(sbd.is_categorical)

    def __repr__(self):
        return "categorical()"


def categorical():
    return Categorical()


class String(Filter):
    def __init__(self):
        super().__init__(sbd.is_string)

    def __repr__(self):
        return "string()"


def string():
    return String()


class Boolean(Filter):
    def __init__(self):
        super().__init__(sbd.is_bool)

    def __repr__(self):
        return "boolean()"


def boolean():
    return Boolean()
