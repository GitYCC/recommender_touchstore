import pickle

import numpy as np

from .model import BaseModel


class PopularityModel(BaseModel):
    def __init__(self):
        self._rating_avg = dict()

    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None):
        rating_sum = dict()
        count = dict()
        for i, (u, m) in enumerate(user_movie_pair.tolist()):
            rating_sum[m] = rating_sum.get(m, 0) + y[i]
            count[m] = count.get(m, 0) + 1

        for m in rating_sum.keys():
            self._rating_avg[m] = rating_sum[m] / count[m]

        return self

    def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
        vfunc = np.vectorize(lambda x: self._rating_avg.get(x, np.nan))
        return vfunc(user_movie_pair[:, 1])

    @classmethod
    def load(cls, local_dir):
        instance = PopularityModel()
        path_pickle = local_dir / 'model.pkl'
        with open(path_pickle, 'rb') as input_file:
            params = pickle.load(input_file)
            instance._rating_avg = params['_rating_avg']
        return instance

    def save(self, local_dir):
        params = dict()
        params['_rating_avg'] = self._rating_avg

        path_pickle = local_dir / 'model.pkl'
        with open(path_pickle, 'wb') as output_file:
            pickle.dump(params, output_file)
