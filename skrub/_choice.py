import io
from collections.abc import Sequence

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid


class Choice(Sequence):
    def __init__(self, *options):
        if not options:
            raise TypeError("Choice should be given at least one option.")
        self.options_ = list(options)
        self.name_ = None

    def name(self, name):
        self.name_ = name
        return self

    def __getitem__(self, item):
        return self.options_[item]

    def __len__(self):
        return len(self.options_)

    def __iter__(self):
        return iter(self.options_)

    def __repr__(self):
        options_repr = ", ".join(map(repr, self.options_))
        name_repr = "" if self.name_ is None else f".name({self.name_!r})"
        return f"choose({options_repr}){name_repr}"


def repeat_name(choice):
    return choose(*(choice.name_ for opt in choice.options_))


class Placeholder:
    def __init__(self, name=None):
        self.name_ = name

    def __repr__(self):
        if self.name_ is not None:
            return f"<{self.name_}>"
        return "..."


def choose(*options):
    return Choice(*options)


def as_choice(obj):
    if isinstance(obj, Choice):
        return obj
    return choose(obj)


def first(obj):
    if isinstance(obj, Choice):
        return obj.options_[0]
    return obj


def set_params_to_first(estimator):
    estimator = clone(first(estimator))
    while (choice := find_param_choice(estimator)) is not None:
        name, value = choice
        value = first(value)
        estimator.set_params(**{name: value})
    return estimator


def options(obj):
    if isinstance(obj, Choice):
        return obj.options_
    return [obj]


def find_param_choice(obj):
    if not hasattr(obj, "get_params"):
        return None
    params = obj.get_params(deep=True)
    for param_name, param_value in params.items():
        if isinstance(param_value, Choice):
            return param_name, param_value
    return None


def insert_in_dict(d, key, value, idx):
    kv = list(d.items())
    kv.insert(idx, (key, value))
    return dict(kv)


def expand_grid(grid):
    for param_name, param_value in grid.items():
        param_choice = as_choice(param_value)
        param_options = options(param_choice)
        for pos, value in enumerate(param_options):
            if (choice := find_param_choice(value)) is None:
                continue
            subparam_name, subparam_choice = choice
            value = clone(value)
            subparam_id = f"{param_name}__{subparam_name}"
            placeholder_name = (
                subparam_id if (n := subparam_choice.name_) is None else n
            )
            value.set_params(**{subparam_name: Placeholder(placeholder_name)})
            grid_1 = grid.copy()
            grid_1[param_name] = as_choice(value).name(param_choice.name_)
            # insert next to parent choice rather than at the end
            idx = list(grid_1.keys()).index(param_name) + 1
            grid_1 = insert_in_dict(grid_1, subparam_id, subparam_choice, idx)
            rest = param_options[:pos] + param_options[pos + 1 :]
            if not rest:
                return expand_grid(grid_1)
            grid_2 = grid.copy()
            grid_2[param_name] = choose(*rest).name(param_choice.name_)
            return [*expand_grid(grid_1), *expand_grid(grid_2)]
    return [{k: as_choice(v) for k, v in grid.items()}]


def unwrap_choices(grids):
    return [{k: options(v) for k, v in subg.items()} for subg in grids]


def choice_names_for_expanded_grid(grid):
    grid = [{k: repeat_name(v) for k, v in subg.items()} for subg in grid]
    return list(ParameterGrid(grid))


def show_expanded_grid(grid):
    names = choice_names_for_expanded_grid(grid)
    values = list(ParameterGrid(grid))
    return [
        [
            (default if given is None else given, val)
            for (default, given), val in zip(
                subgrid_names.items(), subgrid_values.values()
            )
        ]
        for subgrid_names, subgrid_values in zip(names, values)
    ]


def expanded_grid_description(grid):
    buf = io.StringIO()
    for subgrid in show_expanded_grid(grid):
        prefix = "- "
        for k, v in subgrid:
            buf.write(f"{prefix}{k!r}: {v}\n")
            prefix = "  "
        buf.write("\n")
    return buf.getvalue()


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
        buf.write("\n")
    return buf.getvalue()
