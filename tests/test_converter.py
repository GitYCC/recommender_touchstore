import numpy as np

import converters


class TestNoContentConverter:
    def test_convert(self, datagroup):
        conv = converters.NoContentConverter()
        user_movie_pair, y, user_feature, movie_feature = conv.convert(datagroup)

        expect_user_movie_pair = np.array([[1, 101], [2, 102], [3, 103]])
        expect_y = np.array([5, 4, 3])
        expect_user_feature = None
        expect_movie_feature = None

        np.testing.assert_array_equal(user_movie_pair, expect_user_movie_pair)
        np.testing.assert_array_equal(y, expect_y)
        assert user_feature == expect_user_feature
        assert movie_feature == expect_movie_feature
