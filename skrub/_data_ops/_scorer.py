import copy
from functools import partial

from sklearn.metrics import check_scoring

from ._data_ops import DataOp, Score
from ._evaluation import evaluate, find_node
from ._utils import unique_renaming


def _prepare_scorer(estimator, scorer_info):
    scorer = check_scoring(estimator, scorer_info["scoring"])
    kwargs = scorer_info["kwargs"] or {}
    if not hasattr(scorer, "get_metadata_routing"):
        return partial(scorer, **kwargs)
    scorer = copy.deepcopy(scorer)
    if hasattr(scorer, "_kwargs"):
        scorer._kwargs = {**scorer._kwargs, **kwargs}
    elif hasattr(scorer, "_scorers"):
        for sub_scorer in scorer._scorers.values():
            sub_scorer._kwargs = {**sub_scorer._kwargs, **kwargs}
    return scorer


def _process_scores(scorer_info, scorer_output):
    name = scorer_info["name"]
    scoring = scorer_info["scoring"]
    if isinstance(scoring, str):
        return [(name or scoring, scorer_output)]
    if isinstance(scorer_output, dict):
        prefix = f"{name}_" if name else ""
        return [(f"{prefix}{k}", v) for (k, v) in scorer_output.items()]
    if not name:
        try:
            name = scoring._score_func.__name__
        except AttributeError:
            try:
                name = scoring.__name__
            except AttributeError:
                name = "score"
    return [(name, scorer_output)]


class Scorer:
    def __call__(self, estimator, X, y):
        score_node = find_node(
            estimator.data_op,
            lambda o: isinstance(o, DataOp) and isinstance(o._skrub_impl, Score),
        )
        if score_node is None:
            scorer = check_scoring(estimator, scoring=None)
            return scorer(estimator, X, y)
        env = estimator._get_env(X, y)
        scorers = evaluate(
            score_node._skrub_impl.scorers, mode="fit_transform", environment=env
        )
        all_scores = []
        for scorer_info in scorers:
            scorer = _prepare_scorer(estimator, scorer_info)
            scorer_output = scorer(estimator, X, y)
            all_scores.extend(_process_scores(scorer_info, scorer_output))
        rename = unique_renaming()
        return {rename(name): score for name, score in all_scores}
