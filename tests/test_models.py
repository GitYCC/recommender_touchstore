import sys

import pytest
import numpy as np

from models.base import BaseModel


class TestBaseModel:

    class MockModel(BaseModel):
        """MockModel implements BaseModel."""

        def fit(self, user_movie_pair, y,
                user_feature=None, movie_feature=None, sample_weight=None):
            pass

        def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
            return np.sum(user_movie_pair, axis=1) % 3

    @pytest.fixture
    def users(self):
        return [1, 2, 3]

    @pytest.fixture
    def movies(self):
        return [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    @pytest.mark.parametrize('recommended_type', ['movie', 'user'])
    def test_recommend(self, recommended_type, users, movies):
        model = self.MockModel()
        result = model.recommend(recommended_type, users, movies)

        target_dict = {(u, m): (u+m) % 3 for u in users for m in movies}
        if recommended_type == 'movie':
            for i, sample in enumerate(result.tolist()):
                target_keep = sys.maxsize
                for m in sample:
                    r = target_dict[(users[i], m)]
                    assert target_keep >= r
                    target_keep = r
        elif recommended_type == 'user':
            for i, sample in enumerate(result.tolist()):
                target_keep = sys.maxsize
                for u in sample:
                    r = target_dict[(u, movies[i])]
                    assert target_keep >= r
                    target_keep = r
