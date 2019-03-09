import numpy as np
from sklearn.metrics import average_precision_score

import config


class RatingEvaluator:

    def __init__(self, user_movie_pair, actu_y, pred_y):
        """Convert datagroup to the format for model fitting.

        Args:
            user_movie_pair (numpy.array, shape (n_samples, 2)):
                Pair of userId and movieId, where n_samples is the number of samples.
            actu_y (numpy.array, shape (n_samples,)):
                Actual target relative to user_movie_pair.
            pred_y (numpy.array, shape (n_samples,)):
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


class RecommendEvaluator:

    def __init__(self, target, actions, recommendations, rec_scores):

        self._target = target
        self._actions = actions
        self._recommendations = recommendations
        self._rec_scores = rec_scores

    def get_mean_average_precision(self, size=None):
        if size is None:
            size = self._recommendations.shape[1]

        num_target = len(self._target)

        selected_recs = self._recommendations[:, 0:size]
        selected_rec_scores = self._rec_scores[:, 0:size]

        true_scores = np.zeros(selected_rec_scores.shape, dtype='float64')
        for i in range(num_target):
            label = np.isin(selected_recs[i, :], self._actions[i])
            true_scores[i, label] = 1

        tot_ap = 0.0
        for i in range(num_target):
            y_true = true_scores[i, :]
            y_score = selected_rec_scores[i, :]
            if np.sum(y_true) < config.FLOAT_EPSILN:
                ap = 0.0
            else:
                ap = average_precision_score(y_true, y_score)
            tot_ap += ap
        mean_ap = tot_ap / num_target

        return mean_ap
