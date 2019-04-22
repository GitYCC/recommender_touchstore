import sys

import pytest
import numpy as np

from models.model import BaseModel
from models import PopularityModel


class TestBaseModel:

    class MockModel(BaseModel):
        """MockModel implements BaseModel."""

        def fit(self, user_movie_pair, y,
                user_feature=None, movie_feature=None, **model_params):
            pass

        def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
            return np.sum(user_movie_pair, axis=1) % 3

        @classmethod
        def load(cls, local_dir):
            pass

        def save(self, local_dir):
            pass

    @pytest.fixture
    def users(self):
        return [1, 2, 3, 4]

    @pytest.fixture
    def movies(self):
        return [11, 12, 13, 14, 15]

    @pytest.mark.parametrize('recommended_type, maxsize', [('movie', None), ('user', None),
                                                           ('movie', 3), ('user', 3)])
    def test_recommend__mock__right_order(self, users, movies, recommended_type, maxsize):
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


class TestPopularityModel:
    case1_movie_popularity = {100: 3., 101: 1., 102: 3., 103: 2., 104: 4.5, 105: 5.}

    def test_fit__use_case1__right_rating_avg(self, user_movie_pair_1, ratings_1):
        model = PopularityModel()

        model = model.fit(user_movie_pair_1, ratings_1)

        for movie in [100, 101, 102, 103, 104, 105]:
            assert model._rating_avg[movie] == self.case1_movie_popularity[movie]

    def test_predict__use_case1_predict_opp__right_ratings(
            self, user_movie_pair_1, ratings_1, user_movie_pair_1_opp):
        model = PopularityModel()
        model.fit(user_movie_pair_1, ratings_1)

        pred = model.predict(user_movie_pair_1_opp)

        expected_pred = np.array(
            [self.case1_movie_popularity[movie] for _, movie in user_movie_pair_1_opp.tolist()])
        np.testing.assert_array_equal(pred, expected_pred)

    def test_recommend__use_case1__right_items_and_scores(self, user_movie_pair_1, ratings_1):
        model = PopularityModel()
        model.fit(user_movie_pair_1, ratings_1)

        users = [0, 1, 2, 3, 4]
        expected_movie_order = [105, 104, 102, 103, 101]
        rec_items, rec_scores = model.recommend(
            recommended_type='movie',
            users=users,
            movies=sorted(expected_movie_order))

        expected_rec_items = np.array([expected_movie_order for _ in range(len(users))])
        expected_rec_scores = np.array([[5., 4.5, 3., 2., 1.] for _ in range(len(users))])
        np.testing.assert_array_equal(rec_items, expected_rec_items)
        np.testing.assert_array_equal(rec_scores, expected_rec_scores)

    def test_save_and_load__do__restore(self, tmpdir):
        model = PopularityModel()
        model._rating_avg = {1: 0.2, 2: 0.5}
        model.save(tmpdir)
        reloaded_model = PopularityModel.load(tmpdir)
        assert reloaded_model._rating_avg == model._rating_avg
