import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors


class Matching(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, aux, main):
        del main
        self.aux_ = aux
        self.neighbors_ = NearestNeighbors(n_neighbors=1).fit(aux)
        return self

    def match(self, main, max_dist):
        distances, indices = self.neighbors_.kneighbors(main, return_distance=True)
        distances, indices = distances.ravel(), indices.ravel()
        reference_distances = self._get_reference_distances(main, indices, distances)
        rescaled_distances = self._rescale_distances(distances, reference_distances)
        return {
            "index": indices,
            "distance": distances,
            "rescaled_distance": rescaled_distances,
            "match_accepted": rescaled_distances <= max_dist,
        }

    def _get_reference_distances(self, main, indices, distances):
        del main, indices, distances
        return 1.0

    def _rescale_distances(self, distances, reference_distances):
        reference_distances = np.asarray(reference_distances)
        ref_is_zero = reference_distances == 0.0
        rescaled_distances = np.zeros_like(distances)
        rescaled_distances[~ref_is_zero] = (
            distances[~ref_is_zero] / reference_distances[~ref_is_zero]
        )
        rescaled_distances[ref_is_zero] = np.inf
        rescaled_distances[distances == 0] = 0.0
        return rescaled_distances


def _sample_pairs(n, n_pairs, random_state):
    assert n > 1
    assert n_pairs > 0
    rng = np.random.default_rng(random_state)
    parts = []
    n_found = 0
    while n_found < n_pairs:
        new_part = rng.integers(n, size=(n_pairs, 2))
        new_part = new_part[new_part[:, 0] != new_part[:, 1]]
        parts.append(new_part)
        n_found += new_part.shape[0]
    return np.concatenate(parts, axis=0)[:n_pairs]


class Percentile(Matching):
    def __init__(self, n_sampled_pairs=500):
        self.n_sampled_pairs = n_sampled_pairs

    def _get_reference_distances(self, main, indices, distances):
        del main, indices, distances
        n_rows = self.aux_.shape[0]


class TargetNeighbor(Matching):
    def __init__(self, reference_neighbor=1):
        self.reference_neighbor = reference_neighbor

    def _get_reference_distances(self, main, distances, indices):
        del main, distances
        reference_distances, _ = self.neighbors_.kneighbors(
            self.aux_[indices],
            return_distance=True,
            n_neighbors=self.reference_neighbor + 1,
        )
        reference_distances = reference_distances[:, -1]
        return reference_distances


class QueryNeighbor(Matching):
    def __init__(self, reference_neighbor=1):
        self.reference_neighbor = reference_neighbor

    def _get_reference_distances(self, main, indices, distances):
        del indices, distances
        reference_distances, _ = self.neighbors_.kneighbors(
            main, return_distance=True, n_neighbors=self.reference_neighbor + 1
        )
        reference_distances = reference_distances[:, -1]
        return reference_distances


class MaxDist(Matching):
    def _get_reference_distances(self, main, indices, distances):
        del main, indices
        return distances.max()
