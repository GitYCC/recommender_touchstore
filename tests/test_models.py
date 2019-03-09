import sys

import pytest
import numpy as np

from models.model import BaseModel
from models import AverageModel


class TestBaseModel:

    class MockModel(BaseModel):
        """MockModel implements BaseModel."""

        def fit(self, user_movie_pair, y,
                user_feature=None, movie_feature=None, sample_weight=None):
            pass

        def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
            return np.sum(user_movie_pair, axis=1) % 3

        def _get_params(self):
            return dict(paraA=self.paraA, paraB=self.paraB)

    @pytest.fixture
    def users(self):
        return [1, 2, 3, 4]

    @pytest.fixture
    def movies(self):
        return [11, 12, 13, 14, 15]

    @pytest.mark.parametrize('recommended_type, maxsize', [('movie', None), ('user', None),
                                                           ('movie', 3), ('user', 3)])
    def test_recommend(self, users, movies, recommended_type, maxsize):
        model = self.MockModel()
        rec_items, rec_scores = model.recommend(recommended_type, users, movies, maxsize=maxsize)

        target_dict = {(u, m): (u+m) % 3 for u in users for m in movies}
        if recommended_type == 'movie':
            for i, sample in enumerate(rec_items.tolist()):
                target_keep = sys.maxsize
                for j, m in enumerate(sample):
                    r = target_dict[(users[i], m)]
                    assert rec_scores[i, j] == r
                    assert target_keep >= r
                    target_keep = r
        elif recommended_type == 'user':
            for i, sample in enumerate(rec_items.tolist()):
                target_keep = sys.maxsize
                for j, u in enumerate(sample):
                    r = target_dict[(u, movies[i])]
                    assert rec_scores[i, j] == r
                    assert target_keep >= r
                    target_keep = r

    def test_save_and_load(self, tmp_path):
        model = self.MockModel()
        model.paraA = 1
        model.paraB = 2
        output_path = tmp_path / 'model.p'
        model.save(output_path)
        reloaded_model = self.MockModel.load(output_path)
        assert reloaded_model.paraA == model.paraA
        assert reloaded_model.paraB == model.paraB


class TestAverageModel:

    def test_fix(self, user_movie_pair_1, ratings_1):
        model = AverageModel()
        model.fit(user_movie_pair_1, ratings_1)
        assert model._weighted_rating_avg[101] == 5.
        assert model._weighted_rating_avg[102] == 1.5
        assert model._weighted_rating_avg[103] == 3.5

    def test_predict(self, user_movie_pair_1, ratings_1, user_movie_pair_2):
        model = AverageModel()
        model.fit(user_movie_pair_1, ratings_1)
        pred = model.predict(user_movie_pair_2)
        expect_pred = np.array([1.5, 5., 3.5, 5., np.nan])
        np.testing.assert_array_equal(pred, expect_pred)

    def test_recommend(self, user_movie_pair_1, ratings_1):
        model = AverageModel()
        model.fit(user_movie_pair_1, ratings_1)
        rec_items, rec_scores = model.recommend(
            recommended_type='movie', users=[1, 2, 3], movies=[101, 102, 103])
        expected_rec_items = np.array([[101, 103, 102],
                                       [101, 103, 102],
                                       [101, 103, 102]])
        expected_rec_scores = np.array([[5., 3.5, 1.5],
                                        [5., 3.5, 1.5],
                                        [5., 3.5, 1.5]])
        np.testing.assert_array_equal(rec_items, expected_rec_items)
        np.testing.assert_array_equal(rec_scores, expected_rec_scores)
