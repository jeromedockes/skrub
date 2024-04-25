import dataclasses
import io
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.utils import check_random_state

from . import _utils


def _with_fields(obj, **fields):
    return obj.__class__(
        **({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)} | fields)
    )


@dataclasses.dataclass
class Outcome:
    value: Any
    name: str | None = None
    in_choice: str | None = None

    def __str__(self):
        if self.name is not None:
            return repr(self.name)
        return repr(self.value)


class BaseChoice:
    pass


@dataclasses.dataclass
class Choice(Sequence, BaseChoice):
    outcomes: list[Any]
    name: str | None = None

    def __post_init__(self):
        if not self.outcomes:
            raise TypeError("Choice should be given at least one outcome.")
        self.outcomes = [
            _with_fields(out, in_choice=self.name) for out in self.outcomes
        ]

    def take_outcome(self, idx):
        out = self.outcomes[idx]
        rest = self.outcomes[:idx] + self.outcomes[idx + 1 :]
        if not rest:
            return out, None
        return out, _with_fields(self, outcomes=rest)

    def map_values(self, func):
        outcomes = [out.value(func(out.value)) for out in self.outcomes]
        return _with_fields(self, outcomes=outcomes)

    def __repr__(self):
        if self.outcomes[0].name is None:
            args = [out.value for out in self.outcomes]
        else:
            args = {out.name: out.value for out in self.outcomes}
        args_r = _utils.repr_args(
            (args,),
            {"name": self.name},
            defaults={"name": None},
        )
        return f"choose_from({args_r})"

    def __getitem__(self, item):
        return self.outcomes[item]

    def __len__(self):
        return len(self.outcomes)

    def __iter__(self):
        return iter(self.outcomes)


def choose_from(outcomes, name=None):
    if isinstance(outcomes, Mapping):
        prepared_outcomes = [Outcome(val, key) for key, val in outcomes.items()]
    else:
        prepared_outcomes = [Outcome(val) for val in outcomes]
    return Choice(prepared_outcomes, name=name)


class Optional(Choice):
    def __repr__(self):
        args = _utils.repr_args(
            (unwrap_first(self),), {"name": self.name}, defaults={"name": None}
        )
        return f"optional({args})"


def optional(value, name=None):
    return Optional([Outcome(value, "true"), Outcome(None, "false")], name=name)


def _check_bounds(low, high, log):
    if high < low:
        raise ValueError(
            f"'high' must be greater than 'low', got low={low}, high={high}"
        )
    if log and low <= 0:
        raise ValueError(f"To use log space 'low' must be > 0, got low={low}")


@dataclasses.dataclass
class NumericOutcome(Outcome):
    value: int | float
    is_from_log_scale: bool = False
    name: str | None = None
    in_choice: str | None = None


def _repr_numeric_choice(choice):
    args = _utils.repr_args(
        (choice.low, choice.high),
        {
            "log": choice.log,
            "n_steps": getattr(choice, "n_steps", None),
            "name": choice.name,
        },
        defaults={"log": False, "n_steps": None, "name": None},
    )
    if choice.to_int:
        return f"choose_int({args})"
    return f"choose_float({args})"


@dataclasses.dataclass
class NumericChoice(BaseChoice):
    low: float
    high: float
    log: bool
    to_int: bool
    name: str = None

    def __post_init__(self):
        _check_bounds(self.low, self.high, self.log)
        if self.log:
            self._distrib = stats.loguniform(self.low, self.high)
        else:
            self._distrib = stats.uniform(self.low, self.high)

    def rvs(self, size=None, random_state=None):
        value = self._distrib.rvs(size=size, random_state=random_state)
        if self.to_int:
            value = value.astype(int)
        return NumericOutcome(value, is_from_log_scale=self.log, in_choice=self.name)

    def __repr__(self):
        return _repr_numeric_choice(self)


@dataclasses.dataclass
class DiscretizedNumericChoice(Sequence):
    low: float
    high: float
    n_steps: int
    log: bool
    to_int: bool
    name: str = None

    def __post_init__(self):
        _check_bounds(self.low, self.high, self.log)
        if self.log:
            low, high = np.log(self.low), np.log(self.high)
        else:
            low, high = self.low, self.high
        self.grid = np.linspace(low, high, self.n_steps)
        if self.log:
            self.grid = np.exp(self.grid)
        if self.to_int:
            self.grid = np.round(self.grid).astype(int)

    def rvs(self, size=None, random_state=None):
        random_state = check_random_state(random_state)
        value = random_state.choice(self.grid, size=size)
        return NumericOutcome(value, self.log, in_choice=self.name)

    def __getitem__(self, item):
        return self.grid[item]

    def __len__(self):
        return len(self.grid)

    def __iter__(self):
        return iter(self.grid)

    def __repr__(self):
        return _repr_numeric_choice(self)


def choose_float(low, high, log=False, n_steps=None, name=None):
    if n_steps is None:
        return NumericChoice(low, high, log=log, to_int=False, name=name)
    return DiscretizedNumericChoice(
        low, high, log=log, to_int=False, n_steps=n_steps, name=name
    )


def choose_int(low, high, log=False, n_steps=None, name=None):
    if n_steps is None:
        return NumericChoice(low, high, log=log, to_int=True, name=name)
    return DiscretizedNumericChoice(
        low, high, log=log, to_int=True, n_steps=n_steps, name=name
    )


@dataclasses.dataclass
class Placeholder:
    name: str | None = None

    def __repr__(self):
        if self.name is not None:
            return f"<{self.name}>"
        return "..."


def unwrap_first(obj):
    if isinstance(obj, Choice):
        return obj.outcomes[0].value
    if isinstance(obj, NumericChoice):
        return obj.rvs(random_state=0).value
    if isinstance(obj, Outcome):
        return obj.value
    return obj


def unwrap(obj):
    if isinstance(obj, Choice):
        return [outcome.value for outcome in obj.outcomes]
    if isinstance(obj, Outcome):
        return obj.value
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
        if isinstance(param, Choice) and len(param.outcomes) == 1:
            param = param.outcomes[0]
        if isinstance(param, (Outcome, BaseChoice)):
            new_grid[param_name] = param
        else:
            # In this case we have a 'raw' estimator that has not been wrapped
            # in an Outcome. Therefore it is not part of a choice itself, but it
            # contains a choice. We pull out the choices to include them in the
            # grid, but the param itself does not need to be in the grid so we
            # don't include it to keep the grid more compact.
            param = Outcome(param)
        if isinstance(param, BaseChoice):
            continue
        all_subparam_choices = _find_param_choices(param.value)
        if not all_subparam_choices:
            continue
        placeholders = {}
        for subparam_name, subparam_choice in all_subparam_choices.items():
            subparam_id = f"{param_name}__{subparam_name}"
            placeholder_name = subparam_id if (n := subparam_choice.name) is None else n
            placeholders[subparam_name] = Placeholder(placeholder_name)
            new_grid[subparam_id] = subparam_choice
        if param_name in new_grid:
            estimator = clone(param.value)
            estimator.set_params(**placeholders)
            new_grid[param_name] = _with_fields(param, value=estimator)
    return new_grid


def _split_grid(grid):
    grid = _extract_choices(grid)
    for param_name, param in grid.items():
        if not isinstance(param, Choice):
            continue
        for idx, outcome in enumerate(param.outcomes):
            if _find_param_choices(outcome.value):
                grid_1 = grid.copy()
                grid_1[param_name] = outcome
                _, rest = param.take_outcome(idx)
                if rest is None:
                    return _split_grid(grid_1)
                grid_2 = grid.copy()
                grid_2[param_name] = rest
                return [*_split_grid(grid_1), *_split_grid(grid_2)]
    return [grid]


def _check_name_collisions(subgrid):
    all_names = {}
    for param_id, param in subgrid.items():
        name = param.name or param_id
        if name in all_names:
            raise ValueError(
                f"Parameter alias {name!r} used for "
                f"several parameters: {all_names[name], (param_id, param)}."
            )
        all_names[name] = (param_id, param)


def expand_grid(grid):
    grid = _split_grid(grid)
    new_grid = []
    for subgrid in grid:
        new_subgrid = {}
        for k, v in subgrid.items():
            if isinstance(v, Outcome):
                v = Choice([v], name=v.in_choice)
            new_subgrid[k] = v
        new_grid.append(new_subgrid)
        _check_name_collisions(new_subgrid)
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
            if v.name is not None:
                k = v.name
            if isinstance(v, NumericChoice):
                # no need to repeat the name (already in the key) hence name(None)
                write_indented(
                    f"{prefix}{k!r}: ", f"{_with_fields(v, name=None)}\n", buf
                )
            elif len(v.outcomes) == 1:
                write_indented(f"{prefix}{k!r}: ", f"{v.outcomes[0]}\n", buf)
            else:
                buf.write(f"{prefix}{k!r}:\n")
                for outcome in v.outcomes:
                    write_indented("      - ", f"{outcome}\n", buf)
            prefix = "  "
    return buf.getvalue()


def params_description(grid_entry):
    buf = io.StringIO()
    for param_id, param in grid_entry.items():
        choice_name = param.in_choice or param_id
        value = param.name or param.value
        write_indented(f"{choice_name!r}: ", f"{value!r}\n", buf)
    return buf.getvalue()
