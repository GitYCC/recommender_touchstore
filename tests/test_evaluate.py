import numpy as np

from evaluate import RatingEvaluator, RecommendEvaluator
import config


class TestRatingEvaluator:

    def test_get_rms(self, mocker):
        acut_y = np.array([1., 2., 3.])
        pred_y_1 = np.array([1., 2., 3.])
        pred_y_2 = np.array([3., 2., 1.])

        pair = mocker.MagicMock()
        assert (0. - config.FLOAT_EPSILN < RatingEvaluator(pair, acut_y, pred_y_1).get_rms()
                < 0. + config.FLOAT_EPSILN)
        assert (1.632993 - config.FLOAT_EPSILN <
                RatingEvaluator(pair, acut_y, pred_y_2).get_rms() <
                1.632993 + config.FLOAT_EPSILN)


class TestRecommendEvaluator:
    def test_get_mean_average_precision(self, mocker):
        target = [1, 2, 3]
        actions = [[101], [103, 101], [104]]
        recommendations = np.array([[102, 101, 103, 104],
                                    [103, 102, 101, 104],
                                    [103, 102, 101, 104]])
        rec_scores = np.array([[0.5, 0.4, 0.3, 0.1],
                               [0.8, 0.5, 0.1, 0.1],
                               [0.8, 0.5, 0.1, 0.05]])

        evaluator = RecommendEvaluator(target, actions,  recommendations, rec_scores)

        mean_ap = evaluator.get_mean_average_precision(size=3)
        assert 0.444444 - config.FLOAT_EPSILN < mean_ap < 0.444444 + config.FLOAT_EPSILN

        mean_ap = evaluator.get_mean_average_precision(size=None)
        assert 0.5 - config.FLOAT_EPSILN < mean_ap < 0.5 + config.FLOAT_EPSILN

    def test_get_mean_average_precision__have_nan__right_mean_ap(self, mocker):
        target = [1, 2, 3]
        actions = [[101], [103, 101], [104]]
        recommendations = np.array([[102, 101, 103, np.nan],
                                    [103, 102, 101, np.nan],
                                    [103, 102, 101, np.nan]])
        rec_scores = np.array([[0.5, 0.4, 0.3, np.nan],
                               [0.8, 0.5, 0.1, np.nan],
                               [0.8, 0.5, 0.1, np.nan]])

        evaluator = RecommendEvaluator(target, actions,  recommendations, rec_scores)

        mean_ap = evaluator.get_mean_average_precision(size=None)
        assert 0.444444 - config.FLOAT_EPSILN < mean_ap < 0.444444 + config.FLOAT_EPSILN
