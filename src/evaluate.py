import numpy as np


class Evaluator:

    def __init__(self, model, user_movie_pair, y, user_feature=None, movie_feature=None):
        """Convert datagroup to the format for model fitting.

        Args:
            model (models.base)
            user_movie_pair ({array-like, sparse matrix}, shape (n_samples, 2)):
                Pair of userId and movieId, where n_samples is the number of samples.
            y (array-like, shape (n_samples,)):
                Actual target relative to user_movie_pair.

        """
        self._model = model
        self._user_movie_pair = user_movie_pair
        self._y = y
        self._user_feature = user_feature
        self._movie_feature = movie_feature

        self._data = dict()

    def _get_pred_y(self):
        if 'pred_y' not in self._data:
            pred_y = self._model.predict(self._user_movie_pair,
                                         self._user_feature,
                                         self._movie_feature)
            self._data['pred_y'] = pred_y
        return self._data['pred_y']

    def get_rms(self):
        pred_y = self._get_pred_y()

        ele_square_diff = (self._y - pred_y)**2
        # drop na
        ele_square_diff = ele_square_diff[~np.isnan(ele_square_diff)]

        rms = (np.average(ele_square_diff))**0.5
        return rms
