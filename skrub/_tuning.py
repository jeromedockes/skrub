import io
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.utils import check_random_state

from . import _utils
from ._fluent_classes import fluent_class


@fluent_class
class Outcome:
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
    outcomes_: list[Any]
    name_: str | None = None

    def __post_init__(self):
        if not self.outcomes_:
            raise TypeError("Choice should be given at least one outcome.")
        self.outcomes_ = [out.in_choice(self.name_) for out in self.outcomes_]

    def take_outcome(self, idx):
        out = self.outcomes_[idx]
        rest = self.outcomes_[:idx] + self.outcomes_[idx + 1 :]
        if not rest:
            return out, None
        return out, self._with_params(outcomes=rest)

    def map_values(self, func):
        outcomes = [out.value(func(out.value_)) for out in self.outcomes_]
        return self.outcomes(outcomes)

    def __repr__(self):
        args = [out.value_ for out in self.outcomes_ if out.name_ is None]
        kwargs = {
            out.name_: out.value_ for out in self.outcomes_ if out.name_ is not None
        }
        args_r = _utils.repr_args(args, kwargs)
        return f"choose_from({args_r})" + self._get_setters_snippet()

    def __getitem__(self, item):
        return self.outcomes_[item]

    def __len__(self):
        return len(self.outcomes_)

    def __iter__(self):
        return iter(self.outcomes_)


def _check_bounds(low, high, log):
    if high < low:
        raise ValueError(
            f"'high' must be greater than 'low', got low={low}, high={high}"
        )
    if log and low <= 0:
        raise ValueError(f"To use log space 'low' must be > 0, got low={low}")


@fluent_class
class NumericOutcome(Outcome):
    value_: int | float
    is_from_log_scale_: bool = False
    name_: str | None = None
    in_choice_: str | None = None


@fluent_class
class NumericChoice(BaseChoice):
    low_: float
    high_: float
    log_: bool
    to_int_: bool
    name_: str = None

    def __post_init__(self):
        _check_bounds(self.low_, self.high_, self.log_)
        if self.log_:
            self._distrib = stats.loguniform(self.low_, self.high_)
        else:
            self._distrib = stats.uniform(self.low_, self.high_)

    def rvs(self, size=None, random_state=None):
        value = self._distrib.rvs(size=size, random_state=random_state)
        if self.to_int_:
            value = value.astype(int)
        return NumericOutcome(value, self.log_, in_choice=self.name_)

    def __repr__(self):
        args = _utils.repr_args(
            (self.low_, self.high_),
            {
                "log": self.log_,
                "n_steps": getattr(self, "n_steps_", None),
                "name": self.name_,
            },
            defaults={"log": False, "n_steps": None, "name": None},
        )
        if self.to_int_:
            return f"choose_int({args})"
        return f"choose_float({args})"


@fluent_class
class DiscretizedNumericChoice(Sequence, NumericChoice):
    low_: float
    high_: float
    n_steps_: int
    log_: bool
    to_int_: bool
    name_: str = None

    def __post_init__(self):
        _check_bounds(self.low_, self.high_, self.log_)
        if self.log_:
            low, high = np.log(self.low_), np.log(self.high_)
        else:
            low, high = self.low_, self.high_
        self.grid = np.linspace(low, high, self.n_steps_)
        if self.log_:
            self.grid = np.exp(self.grid)
        if self.to_int_:
            self.grid = np.round(self.grid).astype(int)

    def rvs(self, size=None, random_state=None):
        random_state = check_random_state(random_state)
        value = random_state.choice(self.grid, size=size)
        return NumericOutcome(value, self.log_, in_choice=self.name_)

    def __repr__(self):
        return super().__repr__()

    def __getitem__(self, item):
        return self.grid[item]

    def __len__(self):
        return len(self.grid)

    def __iter__(self):
        return iter(self.grid)


class Optional(Choice):
    def __repr__(self):
        args = _utils.repr_args(
            (unwrap_first(self),), {"name": self.name_}, defaults={"name": None}
        )
        return f"optional({args})"


def choose_from(*outcomes, **named_outcomes):
    prepared_outcomes = [Outcome(outcome) for outcome in outcomes] + [
        Outcome(val, name) for name, val in named_outcomes.items()
    ]
    return Choice(prepared_outcomes)


class Pick(Choice):
    # TODO only one API should be kept, see description of pick_from

    def __repr__(self):
        if self.outcomes_[0].name_ is None:
            args = [out.value_ for out in self.outcomes_]
        else:
            args = {out.name_: out.value_ for out in self.outcomes_}
        args_r = _utils.repr_args(
            (args,),
            {"name": self.name_},
            defaults={"name": None},
        )
        return f"pick_from({args_r})"


def pick_from(outcomes, name=None):
    # TODO either remove or replace choose_from with this the two are included
    # now to facilitate discussions, but in the end only one API should remain,
    # (passing a dict or kwargs), and be called choose_from
    if isinstance(outcomes, Mapping):
        prepared_outcomes = [Outcome(val, key) for key, val in outcomes.items()]
    else:
        prepared_outcomes = [Outcome(val) for val in outcomes]
    return Pick(prepared_outcomes, name=name)


def optional(value, name=None):
    return Optional([Outcome(value, "true"), Outcome(None, "false")], name=name)


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


class Placeholder:
    def __init__(self, name=None):
        self.name_ = name

    def __repr__(self):
        if self.name_ is not None:
            return f"<{self.name_}>"
        return "..."


def unwrap_first(obj):
    if isinstance(obj, Choice):
        return obj.outcomes_[0].value_
    if isinstance(obj, NumericChoice):
        return obj.rvs(random_state=0).value_
    if isinstance(obj, Outcome):
        return obj.value_
    return obj


def unwrap(obj):
    if isinstance(obj, Choice):
        return [outcome.value_ for outcome in obj.outcomes_]
    if isinstance(obj, Outcome):
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
        if isinstance(param, Choice) and len(param.outcomes_) == 1:
            param = param.outcomes_[0]
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
            new_grid[param_name] = param._with_params(value=estimator)
    return new_grid


def _split_grid(grid):
    grid = _extract_choices(grid)
    for param_name, param in grid.items():
        if not isinstance(param, Choice):
            continue
        for idx, outcome in enumerate(param.outcomes_):
            if _find_param_choices(outcome.value_):
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
        name = param.name_ or param_id
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
                v = Choice([v], name=v.in_choice_)
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
            if v.name_ is not None:
                k = v.name_
            if isinstance(v, NumericChoice):
                # no need to repeat the name (already in the key) hence name(None)
                write_indented(f"{prefix}{k!r}: ", f"{v.name(None)}\n", buf)
            elif len(v.outcomes_) == 1:
                write_indented(f"{prefix}{k!r}: ", f"{v.outcomes_[0]}\n", buf)
            else:
                buf.write(f"{prefix}{k!r}:\n")
                for outcome in v.outcomes_:
                    write_indented("      - ", f"{outcome}\n", buf)
            prefix = "  "
    return buf.getvalue()


def params_description(grid_entry):
    buf = io.StringIO()
    for param_id, param in grid_entry.items():
        choice_name = param.in_choice_ or param_id
        value = param.name_ or param.value_
        write_indented(f"{choice_name!r}: ", f"{value!r}\n", buf)
    return buf.getvalue()
