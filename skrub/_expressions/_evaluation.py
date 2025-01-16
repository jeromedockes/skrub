import copy
import functools
import inspect
import types
import warnings
from collections import defaultdict
from types import SimpleNamespace

from sklearn.base import BaseEstimator
from sklearn.base import clone as skl_clone

from .. import _tuning
from ._expressions import (
    _BUILTIN_MAP,
    _BUILTIN_SEQ,
    Expr,
    IfElse,
    Var,
    _Constants,
)
from ._utils import X_NAME, Y_NAME, simple_repr

__all__ = [
    "evaluate",
    "clone",
    "find_X",
    "find_y",
    "find_node_by_name",
    "graph",
    "nodes",
    "clear_results",
    "describe_steps",
    "reachable",
    "get_params",
    "set_params",
]


def _as_gen(f):
    if inspect.isgeneratorfunction(f):
        return f

    @functools.wraps(f)
    def g(*args, **kwargs):
        if False:
            yield
        return f(*args, **kwargs)

    return g


class _ExprTraversal:
    def run(self, expr):
        stack = [expr]
        last_result = None
        while stack:
            top = stack[-1]
            try:
                if inspect.isgenerator(top):
                    stack.append(top.send(last_result))
                    last_result = None
                elif isinstance(top, Expr):
                    if isinstance(top._skrub_impl, IfElse):
                        stack.append(self.handle_if_else(stack.pop()))
                    else:
                        stack.append(self.handle_expr(stack.pop()))
                elif isinstance(top, _BUILTIN_MAP):
                    stack.append(self.handle_mapping(stack.pop()))
                elif isinstance(top, _BUILTIN_SEQ):
                    stack.append(self.handle_seq(stack.pop()))
                elif isinstance(top, slice):
                    stack.append(self.handle_slice(stack.pop()))
                elif isinstance(top, _tuning.BaseChoice):
                    stack.append(self.handle_choice(stack.pop()))
                elif isinstance(top, _tuning.Outcome):
                    stack.append(self.handle_outcome(stack.pop()))
                elif isinstance(top, _tuning.Match):
                    stack.append(self.handle_choice_match(stack.pop()))
                elif isinstance(top, BaseEstimator):
                    stack.append(self.handle_estimator(stack.pop()))
                else:
                    stack.append(self.handle_value(stack.pop()))
            except StopIteration as e:
                last_result = e.value
                stack.pop()
        return last_result

    def handle_if_else(self, expr):
        return (yield from self.handle_expr(expr))

    def handle_expr(self, expr, attributes_to_evaluate=None):
        impl = expr._skrub_impl
        evaluated_attributes = {}
        if attributes_to_evaluate is None:
            attributes_to_evaluate = impl._fields
        for name in attributes_to_evaluate:
            attr = getattr(impl, name)
            evaluated_attributes[name] = yield attr
        return self.compute_result(expr, evaluated_attributes)

    def compute_result(self, expr, evaluated_attributes):
        raise NotImplementedError()

    def handle_estimator(self, estimator):
        params = yield estimator.get_params()
        estimator = skl_clone(estimator)
        estimator.set_params(**params)
        return estimator

    def handle_choice(self, choice):
        if not isinstance(choice, _tuning.Choice):
            # choice is a BaseNumericChoice
            return choice
        new_outcomes = yield choice.outcomes
        return _tuning._with_fields(choice, outcomes=new_outcomes)

    def handle_outcome(self, outcome):
        value = yield outcome.value
        return _tuning._with_fields(outcome, value=value)

    def handle_choice_match(self, choice_match):
        choice = yield choice_match.choice
        mapping = yield choice_match.outcome_mapping
        return _tuning._with_fields(
            choice_match, choice=choice, outcome_mapping=mapping
        )

    @_as_gen
    def handle_value(self, value):
        return value

    def handle_seq(self, seq):
        new_seq = []
        for item in seq:
            value = yield item
            new_seq.append(value)
        return type(seq)(new_seq)

    def handle_mapping(self, mapping):
        new_mapping = {}
        for k, v in mapping.items():
            new_mapping[(yield k)] = yield v
        return type(mapping)(new_mapping)

    def handle_slice(self, s):
        return slice((yield s.start), (yield s.stop), (yield s.step))


class _Evaluator(_ExprTraversal):
    def __init__(self, mode="preview", environment=None, callback=None):
        self.mode = mode
        self.environment = {} if environment is None else environment
        self.callback = callback

    def _pick_mode(self, expr):
        if expr is not self._expr and self.mode != "preview":
            return "fit_transform" if "fit" in self.mode else "transform"
        return self.mode

    def run(self, expr):
        self._expr = expr
        return super().run(expr)

    def handle_if_else(self, expr):
        cond = yield expr._skrub_impl.condition
        if cond:
            return (yield expr._skrub_impl.value_if_true)
        else:
            return (yield expr._skrub_impl.value_if_false)

    def handle_expr(self, expr):
        impl = expr._skrub_impl
        if impl.is_X and X_NAME in self.environment:
            return self.environment[X_NAME]
        if impl.is_y and Y_NAME in self.environment:
            return self.environment[Y_NAME]
        if (
            # if Var, let the usual mechanism fetch the value from the
            # environment and store in results dict. Otherwise override with
            # the provided value.
            not isinstance(impl, Var)
            and impl.name is not None
            and impl.name in self.environment
        ):
            return self.environment[impl.name]
        if self.mode in impl.results:
            return impl.results[self.mode]
        result = yield from super().handle_expr(
            expr, impl.fields_required_for_eval(self._pick_mode(expr))
        )
        impl.results[self.mode] = result
        if self.callback is not None:
            self.callback(expr, result)
        return result

    def handle_choice(self, choice):
        if choice.name is not None and choice.name in self.environment:
            return self.environment[choice.name]
        if self.mode == "preview":
            return (yield _tuning.unwrap_default(choice))
        outcome = choice.chosen_outcome_or_default()
        return (yield outcome)

    def handle_outcome(self, outcome):
        return (yield _tuning.unwrap(outcome))

    def handle_choice_match(self, choice_match):
        outcome = yield choice_match.choice
        return (yield choice_match.outcome_mapping[outcome])

    def compute_result(self, expr, evaluated_attributes):
        mode = self._pick_mode(expr)
        try:
            return expr._skrub_impl.compute(
                SimpleNamespace(**evaluated_attributes),
                mode=mode,
                environment=self.environment,
            )
        except Exception as e:
            expr._skrub_impl.errors[mode] = e
            if mode == "preview":
                raise
            stack = expr._skrub_impl.creation_stack_last_line()
            msg = (
                f"Evaluation of node {expr} failed. See above for full traceback. "
                f"This node was defined here:\n{stack}"
            )
            if hasattr(e, "add_note"):
                e.add_note(msg)
                raise
            # python < 3.11 : we cannot add note to exception so fall back on chaining
            # note this changes the type of exception
            raise RuntimeError(msg) from e


def evaluate(expr, mode="preview", environment=None, callback=None, clear=False):
    if clear:
        clear_results(expr)
    try:
        return _Evaluator(mode=mode, environment=environment, callback=callback).run(
            expr
        )
    finally:
        if clear:
            clear_results(expr)


class _Reachable(_Evaluator):
    def __init__(self, mode):
        self.mode = mode
        self.callback = None

    def run(self, expr):
        self._reachable = {}
        super().run(expr)
        return self._reachable

    def handle_expr(self, expr):
        self._reachable[id(expr)] = expr
        return (yield from super().handle_expr(expr))


def reachable(expr, mode):
    return _Reachable(mode).run(expr)


class _Printer(_ExprTraversal):
    def __init__(self, highlight=False):
        self.highlight = highlight

    def run(self, expr):
        self._seen = set()
        self._lines = []
        self._cache_used = False
        _ = super().run(expr)
        if self._cache_used:
            self._lines.append("* Cached, not recomputed")
        return "\n".join(self._lines)

    def compute_result(self, expr, evaluated_attributes):
        is_seen = id(expr) in self._seen
        open_tag, close_tag = "", ""
        if not is_seen and self.highlight:
            open_tag, close_tag = "\033[1m", "\033[0m"
        line = simple_repr(expr, open_tag, close_tag)
        if is_seen:
            line = f"( {line} )*"
            self._cache_used = True
        self._lines.append(line)
        self._seen.add(id(expr))


def describe_steps(expr, highlight=False):
    return _Printer(highlight=highlight).run(expr)


class _Cloner(_ExprTraversal):
    def __init__(self, replace=None, drop_preview_data=False):
        self.replace = replace
        self.drop_preview_data = drop_preview_data

    def run(self, expr):
        self._replace = {} if self.replace is None else dict(self.replace)
        return super().run(expr)

    def handle_choice(self, choice):
        if id(choice) in self._replace:
            return self._replace[id(choice)]
        new_choice = yield from super().handle_choice(choice)
        if not isinstance(new_choice, _tuning.Choice):
            new_choice = _tuning._with_fields(choice)
        self._replace[id(choice)] = new_choice
        return new_choice

    @_as_gen
    def handle_value(self, value):
        if hasattr(value, "__sklearn_clone__") and not isinstance(
            value.__sklearn_clone__, types.MethodType
        ):
            return copy.deepcopy(value)
        return skl_clone(value, safe=False)

    def compute_result(self, expr, evaluated_attributes):
        if id(expr) in self._replace:
            return self._replace[id(expr)]
        impl = expr._skrub_impl
        clone_impl = impl.__class__(**evaluated_attributes)
        if isinstance(clone_impl, Var) and self.drop_preview_data:
            clone_impl.placeholder = _Constants.NO_VALUE
        clone_impl.is_X = impl.is_X
        clone_impl.is_y = impl.is_y
        clone_impl._creation_stack_lines = impl._creation_stack_lines
        clone_impl.name = impl.name
        clone = Expr(clone_impl)
        self._replace[id(expr)] = clone
        return clone


def clone(expr, replace=None, drop_preview_data=False):
    return _Cloner(replace=replace, drop_preview_data=drop_preview_data).run(expr)


def _unique(seq):
    return list(dict.fromkeys(seq))


def _simplify_graph(graph):
    short = {v: i for i, v in enumerate(graph["nodes"].keys())}
    new_nodes = {short[k]: v for k, v in graph["nodes"].items()}
    new_parents = {
        short[k]: [short[p] for p in _unique(v)] for k, v in graph["parents"].items()
    }
    new_children = {
        short[k]: [short[c] for c in _unique(v)] for k, v in graph["children"].items()
    }
    return {"nodes": new_nodes, "parents": new_parents, "children": new_children}


class _Graph(_ExprTraversal):
    def run(self, expr):
        self._nodes = {}
        self._parents = defaultdict(list)
        self._children = defaultdict(list)
        self._current_expr = []
        _ = super().run(expr)
        graph = {
            "nodes": self._nodes,
            "parents": dict(self._parents),
            "children": dict(self._children),
        }
        return _simplify_graph(graph)

    def handle_expr(self, expr):
        if self._current_expr:
            parent, child = id(expr), id(self._current_expr[-1])
            self._parents[child].append(parent)
            self._children[parent].append(child)
        self._current_expr.append(expr)
        result = yield from super().handle_expr(expr)
        self._current_expr.pop()
        self._nodes[id(expr)] = expr
        return result

    def compute_result(self, expr, evaluated_attributes):
        return expr


def graph(expr):
    return _Graph().run(expr)


def nodes(expr):
    return list(graph(expr)["nodes"].values())


def clear_results(expr):
    for n in nodes(expr):
        n._skrub_impl.results = {}
        n._skrub_impl.errors = {}


def _find_node(expr, predicate):
    for node in nodes(expr):
        if predicate(node):
            return node
    return None


def find_X(expr):
    return _find_node(expr, lambda e: e._skrub_impl.is_X)


def find_y(expr):
    return _find_node(expr, lambda e: e._skrub_impl.is_y)


def find_node_by_name(expr, name):
    return _find_node(expr, lambda e: e._skrub_impl.name == name)


class _ChoiceGraph(_ExprTraversal):
    def run(self, expr):
        self._choices = {}
        self._outcomes = {}
        self._parents = defaultdict(set)
        self._current_outcome = [None]
        _ = super().run(expr)
        short = {v: i for i, v in enumerate(self._choices.keys())}
        self._short_ids = short
        choices = {short[k]: v for k, v in self._choices.items()}
        parents = {k: [short[p] for p in v] for k, v in self._parents.items()}
        # - choices: choice's short id (1, 2, ...) to BaseChoice instance
        # - parents: outcome's id (id(outcome)) to list of its parent choices' short ids
        return {"choices": choices, "parents": parents}

    def handle_choice(self, choice):
        # unlike during evaluation here we need pre-ordering
        self._parents[self._current_outcome[-1]].add(id(choice))
        self._choices[id(choice)] = choice
        yield from super().handle_choice(choice)
        return choice

    def handle_outcome(self, outcome):
        self._current_outcome.append(id(outcome))
        yield from super().handle_outcome(outcome)
        self._current_outcome.pop()
        return outcome

    def handle_choice_match(self, choice_match):
        yield choice_match.choice
        for outcome in choice_match.choice.outcomes:
            value = outcome.value
            self._current_outcome.append(id(outcome))
            yield choice_match.outcome_mapping[value]
            self._current_outcome.pop()
        return choice_match

    def compute_result(self, expr, evaluated_attributes):
        return expr


def choices(expr):
    return _ChoiceGraph().run(expr)["choices"]


def choice_graph(expr):
    full_builder = _ChoiceGraph()
    full_graph = full_builder.run(expr)
    # identify which choices are used before the nodes marked as X or y. Those
    # choices cannot be tuned (they are needed before the cv loop starts) so
    # they will be clamped to a single value (the chosen outcome if that has been
    # set otherwise the default)
    x_y_choices = set()
    for node in [find_X(expr), find_y(expr)]:
        if node is not None:
            node_graph = _ChoiceGraph().run(node)
            x_y_choices.update(
                {
                    # convert from python id to short id
                    full_builder._short_ids[id(c)]
                    for c in node_graph["choices"].values()
                }
            )
    if x_y_choices:
        warnings.warn(
            "The following choices are used in the construction of X or y, "
            "so their value cannot be tuned because they are needed outside "
            "of the cross-validation loop. They will be clamped to their "
            f"default value: {[full_graph['choices'][k] for k in x_y_choices]}"
        )
    full_graph["x_y_choices"] = x_y_choices
    return full_graph


def _expand_grid(graph, grid):
    def choice_range(choice_id):
        # The range of possible values for a choice.
        # for numeric choices it is the object itself and for Choices the range
        # of possible outcome indices.
        # if the choice is used in X or y, it is clamped to a single value
        choice = graph["choices"][choice_id]
        if choice_id in graph["x_y_choices"]:
            if isinstance(choice, _tuning.Choice):
                return [choice.chosen_outcome_idx or 0]
            else:
                return [choice.default()]
        else:
            if isinstance(choice, _tuning.Choice):
                return list(range(len(choice.outcomes)))
            else:
                return choice

    def has_parents(choice_id):
        # if any of the outcomes in a choice contains another choice. in this
        # case it needs to be on a separate subgrid.
        choice = graph["choices"][choice_id]
        if not isinstance(choice, _tuning.Choice):
            return False
        for outcome in choice.outcomes:
            if graph["parents"].get(id(outcome), None):
                return True
        return False

    # extract
    if None not in graph["parents"]:
        return [grid]
    for choice_id in graph["parents"][None]:
        if not has_parents(choice_id):
            grid[choice_id] = choice_range(choice_id)
    # split
    remaining = [c_id for c_id in graph["parents"][None] if c_id not in grid]
    if not remaining:
        return [grid]
    choice_id = remaining[0]
    subgrids = []
    for outcome_idx in choice_range(choice_id):
        outcome = graph["choices"][choice_id].outcomes[outcome_idx]
        new_subgrid = grid.copy()
        graph = graph.copy()
        graph["parents"][None] = graph["parents"].get(id(outcome), []) + remaining[1:]
        new_subgrid[choice_id] = [outcome_idx]
        new_subgrid = _expand_grid(graph, new_subgrid)
        subgrids.extend(new_subgrid)
    return subgrids


def param_grid(expr):
    graph = choice_graph(expr)
    return _expand_grid(graph, {})


def get_params(expr):
    expr_choices = choices(expr)
    params = {}
    for k, v in expr_choices.items():
        if isinstance(v, _tuning.Choice):
            params[k] = getattr(v, "chosen_outcome_idx", None)
        else:
            params[k] = getattr(v, "chosen_outcome", None)
    return params


def set_params(expr, params):
    expr_choices = choices(expr)
    for k, v in params.items():
        target = expr_choices[k]
        if isinstance(target, _tuning.Choice):
            target.chosen_outcome_idx = v
        else:
            target.chosen_outcome = v
