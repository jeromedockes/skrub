import io
from collections.abc import Sequence
from typing import Any

from scipy import stats
from sklearn.base import clone

from ._fluent_classes import fluent_class


@fluent_class
class Option:
    value_: Any
    name_: str | None = None
    in_choice_: str | None = None

    def __str__(self):
        if self.name_ is not None:
            return repr(self.name_)
        return repr(self.value_)


class BaseChoice:
    pass


@fluent_class
class Choice(Sequence, BaseChoice):
    options_: list[Any]
    name_: str | None = None

    def __post_init__(self):
        if not self.options_:
            raise TypeError("Choice should be given at least one option.")
        self.options_ = list(self.options_)
        self._update_option_names()

    def _update_option_names(self):
        for opt in self.options_:
            opt.in_choice_ = self.name_

    def __getitem__(self, item):
        return self.options_[item]

    def __len__(self):
        return len(self.options_)

    def __iter__(self):
        return iter(self.options_)

    def _get_factory_repr(self):
        options_repr = ", ".join(
            [repr(opt.value_) for opt in self.options_ if opt.name_ is None]
            + [
                f"{opt.name_}={opt.value_!r}"
                for opt in self.options_
                if opt.name_ is not None
            ]
        )
        return f"choose({options_repr})"


@fluent_class
class RandomNumber(BaseChoice):
    low_: float
    high_: float
    log_: bool
    to_int_: bool
    name_: str = None

    def __post_init__(self):
        if self.log_:
            self._distrib = stats.loguniform(self.low_, self.high_)
        else:
            self._distrib = stats.uniform(self.low_, self.high_)

    def rvs(self, size=None, random_state=None):
        value = self._distrib.rvs(size=size, random_state=random_state)
        if self.to_int_:
            value = value.astype(int)
        return Option(value, in_choice=self.name_)

    def _get_factory_repr(self):
        parts = [repr(self.low_), repr(self.high_)]
        if self.log_:
            parts.append("log=True")
        args = ", ".join(parts)
        if self.to_int_:
            return f"choose_int({args})"
        return f"choose_float({args})"


class Optional(Choice):
    def _get_factory_repr(self):
        return f"optional({self.options_[0].value_!r})"


def choose(*options, **named_options):
    prepared_options = [Option(opt) for opt in options] + [
        Option(val, name) for name, val in named_options.items()
    ]
    return Choice(prepared_options)


def optional(option):
    return Optional([Option(option, "true"), Option(None, "false")])


def choose_float(low, high, log=False):
    return RandomNumber(low, high, log=log, to_int=False)


def choose_int(low, high, log=False):
    return RandomNumber(low, high, log=log, to_int=True)


class Placeholder:
    def __init__(self, name=None):
        self.name_ = name

    def __repr__(self):
        if self.name_ is not None:
            return f"<{self.name_}>"
        return "..."


def unwrap_first(obj):
    if isinstance(obj, Choice):
        return obj.options_[0].value_
    if isinstance(obj, RandomNumber):
        return obj.rvs(random_state=0).value_
    if isinstance(obj, Option):
        return obj.value_
    return obj


def unwrap(obj):
    if isinstance(obj, Choice):
        return [opt.value_ for opt in obj.options_]
    if isinstance(obj, Option):
        return obj.value_
    return obj


def contains_choice(estimator):
    return isinstance(estimator, Choice) or bool(_find_param_choices(estimator))


def set_params_to_first(estimator):
    estimator = unwrap_first(estimator)
    if not hasattr(estimator, "set_params"):
        return estimator
    estimator = clone(estimator)
    while param_choices := _find_param_choices(estimator):
        params = {k: unwrap_first(v) for k, v in param_choices.items()}
        estimator.set_params(**params)
    return estimator


def _find_param_choices(obj):
    if not hasattr(obj, "get_params"):
        return []
    params = obj.get_params(deep=True)
    return {k: v for k, v in params.items() if isinstance(v, BaseChoice)}


def _extract_choices(grid):
    new_grid = {}
    for param_name, param in grid.items():
        if isinstance(param, Choice) and len(param.options_) == 1:
            param = param.options_[0]
        if isinstance(param, (Option, BaseChoice)):
            new_grid[param_name] = param
        else:
            # In this case we have a 'raw' estimator that has not been wrapped
            # in an Option. Therefore it is not part of a choice itself, but it
            # contains a choice. We pull out the choices to include them in the
            # grid, but the param itself does not need to be in the grid so we
            # don't include it to keep the grid more compact.
            param = Option(param)
        if isinstance(param, BaseChoice):
            continue
        all_subparam_choices = _find_param_choices(param.value_)
        if not all_subparam_choices:
            continue
        placeholders = {}
        for subparam_name, subparam_choice in all_subparam_choices.items():
            subparam_id = f"{param_name}__{subparam_name}"
            placeholder_name = (
                subparam_id if (n := subparam_choice.name_) is None else n
            )
            placeholders[subparam_name] = Placeholder(placeholder_name)
            new_grid[subparam_id] = subparam_choice
        if param_name in new_grid:
            estimator = clone(param.value_)
            estimator.set_params(**placeholders)
            new_grid[param_name] = Option(estimator, param.name_, param.in_choice_)
    return new_grid


def _split_grid(grid):
    grid = _extract_choices(grid)
    for param_name, param in grid.items():
        if not isinstance(param, Choice):
            continue
        for idx, option in enumerate(param.options_):
            if _find_param_choices(option.value_):
                grid_1 = grid.copy()
                grid_1[param_name] = option
                rest = param.options_[:idx] + param.options_[idx + 1 :]
                if not rest:
                    return _split_grid(grid_1)
                grid_2 = grid.copy()
                grid_2[param_name] = Choice(rest, name=param.name_)
                return [*_split_grid(grid_1), *_split_grid(grid_2)]
    return [grid]


def expand_grid(grid):
    grid = _split_grid(grid)
    # wrap all Options in a Choice because ParamGrid wants all values to be
    # iterables.
    new_grid = []
    for subgrid in grid:
        new_subgrid = {}
        for k, v in subgrid.items():
            if isinstance(v, Option):
                v = Choice([v], name=v.in_choice_)
            new_subgrid[k] = v
        new_grid.append(new_subgrid)
    return new_grid


def write_indented(prefix, text, ostream):
    istream = io.StringIO(text)
    ostream.write(prefix)
    ostream.write(next(istream))
    for line in istream:
        ostream.write(" " * len(prefix))
        ostream.write(line)
    return ostream.getvalue()


def grid_description(grid):
    buf = io.StringIO()
    for subgrid in grid:
        prefix = "- "
        for k, v in subgrid.items():
            if v.name_ is not None:
                k = v.name_
            if isinstance(v, RandomNumber):
                write_indented(f"{prefix}{k!r}: ", f"{v._get_factory_repr()}\n", buf)
            elif len(v.options_) == 1:
                write_indented(f"{prefix}{k!r}: ", f"{v.options_[0]}\n", buf)
            else:
                buf.write(f"{prefix}{k!r}:\n")
                for opt in v.options_:
                    write_indented("      - ", f"{opt}\n", buf)
            prefix = "  "
    return buf.getvalue()


def params_description(grid_entry):
    buf = io.StringIO()
    for param_id, param in grid_entry.items():
        choice_name = param.in_choice_ or param_id
        value = param.name_ or param.value_
        write_indented(f"{choice_name!r}: ", f"{value!r}\n", buf)
    return buf.getvalue()
