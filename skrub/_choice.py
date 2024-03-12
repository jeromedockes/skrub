import io
from collections.abc import Sequence

from sklearn.base import clone


class Option:
    def __init__(self, value, name=None, in_choice=None):
        self.value_ = value
        self.name_ = name
        self.in_choice_ = in_choice

    def name(self, new_name):
        self.name_ = new_name

    def in_choice(self, new_in_choice):
        self.in_choice_ = new_in_choice

    def __str__(self):
        if self.name_ is not None:
            return repr(self.name_)
        return repr(self.value_)

    def __repr__(self):
        parts = [f"{self.value_!r}"]
        if self.name_ is not None:
            parts.append(f"name={self.name_!r}")
        if self.in_choice_ is not None:
            parts.append(f"in_choice={self.in_choice_!r}")
        args = ", ".join(parts)
        return f"Option({args})"


class Choice(Sequence):
    def __init__(self, options, name=None):
        if not options:
            raise TypeError("Choice should be given at least one option.")
        self.options_ = list(options)
        self.name_ = name
        self._update_option_names()

    def name(self, name):
        self.name_ = name
        self._update_option_names()
        return self

    def _update_option_names(self):
        for opt in self.options_:
            opt.in_choice_ = self.name_

    def __getitem__(self, item):
        return self.options_[item]

    def __len__(self):
        return len(self.options_)

    def __iter__(self):
        return iter(self.options_)

    def __repr__(self):
        options_repr = ", ".join(
            [repr(opt.value_) for opt in self.options_ if opt.name_ is None]
            + [
                f"{opt.name_}={opt.value_!r}"
                for opt in self.options_
                if opt.name_ is not None
            ]
        )
        name_repr = "" if self.name_ is None else f".name({self.name_!r})"
        return f"choose({options_repr}){name_repr}"


def choose(*options, **named_options):
    prepared_options = [Option(opt) for opt in options] + [
        Option(val, name) for name, val in named_options.items()
    ]
    return Choice(prepared_options)


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
    if isinstance(obj, Option):
        return obj.value_
    return obj


def unwrap(obj):
    if isinstance(obj, Choice):
        return [opt.value_ for opt in obj.options_]
    if isinstance(obj, Option):
        return obj.value_
    return obj


def contains_choice(obj):
    return isinstance(obj, Choice) or bool(_find_param_choices(obj))


def set_params_to_first(estimator):
    estimator = clone(unwrap_first(estimator))
    while param_choices := _find_param_choices(estimator):
        params = {k: unwrap_first(v) for k, v in param_choices.items()}
        estimator.set_params(**params)
    return estimator


def _find_param_choices(obj):
    if not hasattr(obj, "get_params"):
        return []
    params = obj.get_params(deep=True)
    return {k: v for k, v in params.items() if isinstance(v, Choice)}


def _extract_choices(grid):
    new_grid = {}
    for param_name, param in grid.items():
        if isinstance(param, Choice) and len(param.options_) == 1:
            param = param.options_[0]
        if isinstance(param, (Option, Choice)):
            new_grid[param_name] = param
        else:
            # In this case we have a 'raw' estimator that has not been wrapped
            # in an Option. Therefore it is not part of a choice itself, but it
            # contains a choice. We pull out the choices to include them in the
            # grid, but the param itself does not need to be in the grid so we
            # don't include it to keep the grid more compact.
            param = Option(param)
        if isinstance(param, Choice):
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
    # wrap all values in a Choice because ParamGrid wants all values to be
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


def grid_description(grid):
    buf = io.StringIO()
    for subgrid in grid:
        prefix = "- "
        for k, v in subgrid.items():
            if v.name_ is not None:
                k = v.name_
            if len(v.options_) == 1:
                buf.write(f"{prefix}{k!r}: {v.options_[0]}\n")
            else:
                buf.write(f"{prefix}{k!r}:\n")
                for opt in v.options_:
                    buf.write(f"      - {opt}\n")
            prefix = "  "
    return buf.getvalue()


def params_description(grid_entry):
    buf = io.StringIO()
    for param_id, param in grid_entry.items():
        choice_name = param.in_choice_ or param_id
        value = param.name_ or param.value_
        buf.write(f"{choice_name!r}: {value!r}\n")
    return buf.getvalue()
