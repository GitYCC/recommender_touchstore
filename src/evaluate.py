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

    def __init__(self, targets, actions, recommendations, rec_scores):

        self._targets = targets
        self._actions = actions
        self._recommendations = recommendations
        self._rec_scores = rec_scores

    def get_mean_average_precision(self, size=None):
        rec_list = [[i for i in sub if not np.isnan(i)] for sub in self._recommendations.tolist()]
        rec_scores_list = \
            [[i for i in sub if not np.isnan(i)] for sub in self._rec_scores.tolist()]

        tot_ap = 0.0
        for target, action, rec, rec_score \
                in zip(self._targets, self._actions, rec_list, rec_scores_list):
            assert len(rec) == len(rec_score)
            rec_true = [(1 if i in action else 0) for i in rec]

            if size:
                rec_true = rec_true[:size]
                rec_score = rec_score[:size]

            if np.sum(rec_true) < config.FLOAT_EPSILN:
                ap = 0.0
            else:
                ap = average_precision_score(rec_true, rec_score)
            tot_ap += ap

        mean_ap = tot_ap / len(self._targets)

        return mean_ap
