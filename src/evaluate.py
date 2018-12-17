import numpy as np


class RatingEvaluator:

    def __init__(self, user_movie_pair, actu_y, pred_y):
        """Convert datagroup to the format for model fitting.

        Args:
            user_movie_pair ({array-like, sparse matrix}, shape (n_samples, 2)):
                Pair of userId and movieId, where n_samples is the number of samples.
            actu_y (array-like, shape (n_samples,)):
                Actual target relative to user_movie_pair.
            pred_y (array-like, shape (n_samples,)):
                Predicted target relative to user_movie_pair.

        """
        self._user_movie_pair = user_movie_pair
        self._actu_y = actu_y
        self._pred_y = pred_y

    def get_rms(self):
        ele_square_diff = (self._actu_y - self._pred_y)**2
        # drop na
        ele_square_diff = ele_square_diff[~np.isnan(ele_square_diff)]

        rms = (np.average(ele_square_diff))**0.5
        return rms
