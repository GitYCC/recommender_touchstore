import pickle

import numpy as np

from .model import BaseModel


class PopularityModel(BaseModel):
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

    @classmethod
    def load(cls, local_dir):
        instance = object.__new__(cls)
        path_pickle = local_dir / 'model.pkl'
        with open(path_pickle, 'rb') as input_file:
            params = pickle.load(input_file)
            for param_name, param_val in params.items():
                setattr(instance, param_name, param_val)
        return instance

    def save(self, local_dir):
        params = dict()
        params['_weighted_rating_avg'] = self._weighted_rating_avg

        path_pickle = local_dir / 'model.pkl'
        with open(path_pickle, 'wb') as output_file:
            pickle.dump(params, output_file)
