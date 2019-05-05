import numpy as np
import pandas as pd

import decorators


class TestNegativeDataDecorator:

    def test_decorate(self, datagroup):
        ratio = 1.0
        negative_val = 0

        decorator = decorators.NegativeDataDecorator()
        decorated_datagroup = decorator.decorate(
            datagroup, negative_data_ratio=ratio, negative_data_value=negative_val)
        count_likes = datagroup.likes.shape[0]

        assert decorated_datagroup.likes.shape[0] == count_likes * (1. + ratio)
        assert np.sum(decorated_datagroup.likes.like == negative_val) == count_likes
        if datagroup.user_feature is not None:
            pd.testing.assert_frame_equal(decorated_datagroup.user_feature, datagroup.user_feature)
        else:
            assert decorated_datagroup.user_feature is None
        if datagroup.movie_feature is not None:
            pd.testing.assert_frame_equal(
                decorated_datagroup.movie_feature, datagroup.movie_feature)
        else:
            assert decorated_datagroup.movie_feature is None
