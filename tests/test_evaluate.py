import numpy as np

from evaluate import Evaluator
import config


class TestEvaluator:
    def test_get_rms(self, mocker, datagroup):
        acut_y = np.array([1., 2., 3.])
        pred_y_1 = np.array([1., 2., 3.])
        pred_y_2 = np.array([3., 2., 1.])
        model = mocker.MagicMock()
        model.predict.return_value = pred_y_1
        pair = mocker.MagicMock()
        assert (-1 * config.FLOAT_EPSILN < Evaluator(model, pair, acut_y).get_rms()
                < config.FLOAT_EPSILN)
        model.predict.return_value = pred_y_2
        assert (1.632993 - 1 * config.FLOAT_EPSILN < Evaluator(model, pair, acut_y).get_rms()
                < 1.632993 + config.FLOAT_EPSILN)
