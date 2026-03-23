from sklearn.metrics import check_scoring

from ._data_ops import DataOp, Score
from ._evaluation import evaluate, find_node
from ._utils import unique_renaming


def _process_scores(scorer_info, scorer_output):
    if scorer_info["name"]:
        prefix = scorer_info["name"] + "_"
    else:
        prefix = ""
    scoring = scorer_info["scoring"]
    if isinstance(scoring, str):
        return [(f"{prefix}{scoring}", scorer_output)]
    if isinstance(scorer_output, dict):
        return [(f"{prefix}{k}", v) for (k, v) in scorer_output.items()]
    name = scorer_info["name"]
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
        # - find the Score node if there is one
        #   - scoring passed to cross_validate is added, or replaces?
        score_node = find_node(
            estimator.data_op,
            lambda o: isinstance(o, DataOp) and isinstance(o._skrub_impl, Score),
        )
        env = estimator._get_env(X, y)
        scorers = evaluate(
            score_node._skrub_impl.scorers, mode="fit_transform", environment=env
        )
        all_scores = []
        for scorer_info in scorers:
            scorer = check_scoring(estimator, scorer_info["scoring"])
            kwargs = scorer_info["kwargs"] or {}
            scorer_output = scorer(estimator, X, y, **kwargs)
            all_scores.extend(_process_scores(scorer_info, scorer_output))
        rename = unique_renaming()
        return {rename(name): score for name, score in all_scores}
