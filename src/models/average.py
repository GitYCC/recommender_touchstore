import numpy as np

from .model import BaseModel


class AverageModel(BaseModel):
    def __init__(self):
        self._weighted_rating_avg = dict()

    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None, sample_weight=None):
        n_samples = y.shape[0]
        if sample_weight is None:
            sample_weight = np.full((n_samples, ), 1.0)

        weighted_rating_sum = dict()
        weighted_sum = dict()
        for i, (u, m) in enumerate(user_movie_pair.tolist()):
            weighted_rating_sum[m] = weighted_rating_sum.get(m, 0) + y[i] * sample_weight[i]
            weighted_sum[m] = weighted_sum.get(m, 0) + sample_weight[i]

        for m in weighted_rating_sum.keys():
            self._weighted_rating_avg[m] = weighted_rating_sum[m] / weighted_sum[m]

        return self

    def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
        vfunc = np.vectorize(lambda x: self._weighted_rating_avg.get(x, np.nan))
        return vfunc(user_movie_pair[:, 1])

    def _get_params(self):
        params = dict()
        params['_weighted_rating_avg'] = self._weighted_rating_avg
        return params
