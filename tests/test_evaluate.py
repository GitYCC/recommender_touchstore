import numpy as np

from evaluate import RatingEvaluator
import config


class TestEvaluator:
    def test_get_rms(self, mocker, datagroup):
        acut_y = np.array([1., 2., 3.])
        pred_y_1 = np.array([1., 2., 3.])
        pred_y_2 = np.array([3., 2., 1.])

        pair = mocker.MagicMock()
        assert (-1 * config.FLOAT_EPSILN < RatingEvaluator(pair, acut_y, pred_y_1).get_rms()
                < config.FLOAT_EPSILN)
        assert (1.632993 - 1 * config.FLOAT_EPSILN <
                RatingEvaluator(pair, acut_y, pred_y_2).get_rms() <
                1.632993 + config.FLOAT_EPSILN)
