import sys
import os

import pytest
import numpy as np
import pandas as pd

import config
from models.model import BaseModel
from models import PopularityModel
from models import ItemCosineSimilarity
from models.factorization import LIBMFConnecter
from models import RealValuedMatrixFactorization

ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]


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
        return list(range(200))

    @pytest.fixture
    def movies(self):
        return list(range(1000, 1200))

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

        df = model._df_rating_avg
        for movie in [100, 101, 102, 103, 104, 105]:
            assert df.loc[df.movieId == movie, 'y'].values == self.case1_movie_popularity[movie]

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
        model._df_rating_avg = pd.DataFrame(
            [(1, 0.2), (2, 0.5)],
            columns=['movieId', 'y'],
        )
        model.save(tmpdir)
        reloaded_model = PopularityModel.load(tmpdir)
        assert reloaded_model._rating_avg == model._rating_avg


class TestCosineSimilarity:

    def test_fit__use_case1__right_center_y(self, user_movie_pair_1, ratings_1):
        model = ItemCosineSimilarity()

        model = model.fit(user_movie_pair_1, ratings_1)

        expected_center_y_list = \
            [[100, 3.], [101, 1.], [102, 3.], [103, 2.], [104, 4.5], [105, 5.]]

        assert sorted(model._df_center_y.values.tolist()) == sorted(expected_center_y_list)

    def test_predict__use_case1_predict_opp__right_ratings(
            self, user_movie_pair_1, ratings_1, user_movie_pair_1_opp):
        model = ItemCosineSimilarity()
        model.fit(user_movie_pair_1, ratings_1)

        pred = model.predict(user_movie_pair_1_opp)

        expected_pred_list = [4., 5.5, 5., 1., 3., 5., 1., 4.5, 5., 3.75, 1., 2.]

        assert pred.tolist() == expected_pred_list

    def test_save_and_load__do__restore(self, tmpdir):
        model = ItemCosineSimilarity()
        model._df_ratings = pd.DataFrame(
            [[100, 3.], [101, 1.], [102, 3.], [103, 2.], [104, 4.5], [105, 5.]],
            columns=['movieId', 'center_y'],
        )
        model._df_center_y = pd.DataFrame(
            [[1, 100, 2.], [2, 102, 3.], [3, 105, 5.]],
            columns=['userId', 'movieId', 'rating'],
        )
        model.save(tmpdir)
        reloaded_model = ItemCosineSimilarity.load(tmpdir)

        pd.testing.assert_frame_equal(reloaded_model._df_ratings, model._df_ratings)
        pd.testing.assert_frame_equal(reloaded_model._df_center_y, model._df_center_y)


class TestLIBMFConnecter:

    def test_save_matrix__save_and_load_matrix__restore(self, tmpdir):
        df = pd.DataFrame(
            [[1, 100, 3.], [1, 101, 1.], [2, 102, 3.], [2, 103, 2.]],
            columns=['userId', 'movieId', 'y'],
        )
        path = str(tmpdir / 'matrix.txt')
        index_df, df_user_index, df_movie_index = LIBMFConnecter.save_matrix(df, path)
        restored_df = LIBMFConnecter.load_matrix(path, df_user_index, df_movie_index)

        pd.testing.assert_frame_equal(restored_df, df)

    def test_save_matrix_with_indexer(self, tmpdir):
        path = str(tmpdir / 'matrix.txt')
        df_user_index = pd.DataFrame([(0, 10), (1, 20), (2, 30)], columns=['user_index', 'user'])
        df_item_index = pd.DataFrame([(0, 100), (1, 200)], columns=['item_index', 'item'])
        df = pd.DataFrame(
            [[10, 100, 1.], [10, 200, 2.], [20, 100, 3.], [20, 300, 4.], [40, 100, 5.]],
            columns=['user', 'item', 'y'],
        )
        LIBMFConnecter.save_matrix_with_indexer(
            df, path, df_user_index, df_item_index,
            user_col='user', item_col='item', rating_col='y')
        restored_df = LIBMFConnecter.load_matrix(path, df_user_index, df_item_index,
                                                 user_col='user', item_col='item', rating_col='y')
        expected_df = pd.DataFrame(
            [[10, 100, 1.], [10, 200, 2.], [20, 100, 3.]],
            columns=['user', 'item', 'y'],
        )
        pd.testing.assert_frame_equal(
            restored_df.sort_values(by=['y']).reset_index(drop=True),
            expected_df.sort_values(by=['y']).reset_index(drop=True))

    def test_load_model__given_fixture__restore(self):
        path = os.path.join(ROOT_DIR, 'fixtures', 'libmf_model.txt')
        df_user_index = pd.DataFrame([(0, 10), (1, 20), (2, 30)], columns=['user_index', 'user'])
        df_item_index = pd.DataFrame([(0, 100), (1, 200)], columns=['item_index', 'item'])
        df_user_vector, df_item_vector, user_dim, item_dim, dim, global_b = \
            LIBMFConnecter.load_model(
                path, df_user_index, df_item_index,
                user_col='user', item_col='item'
            )

        expected_df_user_vector = pd.DataFrame(
            [
                (10, np.array([1., 3., 5.])),
                (20, np.array([np.nan, np.nan, np.nan])),
                (30, np.array([2., 4., 6.])),
            ],
            columns=['user', 'vector']
        )
        expected_df_item_vector = pd.DataFrame(
            [
                (100, np.array([-1., -3., -5.])),
                (200, np.array([-2., -4., -6.])),
            ],
            columns=['item', 'vector']
        )
        pd.testing.assert_frame_equal(df_user_vector, expected_df_user_vector)
        pd.testing.assert_frame_equal(df_item_vector, expected_df_item_vector)
        assert user_dim == 3
        assert item_dim == 2
        assert dim == 3
        assert global_b == 0.5

    def test_train__run__no_exception(self, tmpdir):
        log_path = str(tmpdir / 'log.txt')
        model_path = str(tmpdir / 'model.txt')
        matrix_path = os.path.join(ROOT_DIR, 'fixtures', 'libmf_matrix.txt')
        LIBMFConnecter.train(method='RVMF', dim=3, epoch=3, lr=0.001,
                             pth_train=matrix_path, pth_model=model_path, pth_log=log_path,
                             pth_valid=None, l1=0.0, l2=0.0)
        LIBMFConnecter.train(method='OCMF', dim=3, epoch=3, lr=0.001,
                             pth_train=matrix_path, pth_model=model_path, pth_log=log_path,
                             pth_valid=matrix_path, l1=0.0, l2=0.0)


class TestRealValuedMatrixFactorization:

    def test_fit__use_case1__right_global_b(self, user_movie_pair_1, ratings_1):
        model = RealValuedMatrixFactorization()

        model = model.fit(user_movie_pair_1, ratings_1, dim=3, epoch=10, lr=0.1, l1=0.0, l2=0.0)

        expected_global_b = 3.083333
        assert (expected_global_b - config.FLOAT_EPSILN <
                model._global_b < expected_global_b + config.FLOAT_EPSILN)

    @pytest.fixture
    def _full_matrix(self):
        user_movie_pair = np.array([[0, 100], [0, 101], [1, 100], [1, 101]])
        ratings = np.array([1, 2, 3, 4])
        return user_movie_pair, ratings

    @pytest.fixture
    def _non_full_matrix(self):
        user_movie_pair = np.array([[0, 100], [0, 101], [1, 100]])
        ratings = np.array([1, 2, 3])
        return user_movie_pair, ratings

    def test_fit__full_matrix__restore_ratings(self, _full_matrix):
        user_movie_pair, ratings = _full_matrix

        model = RealValuedMatrixFactorization()
        model = model.fit(user_movie_pair, ratings, dim=3, epoch=300, lr=0.1, l1=0.0, l2=0.0)

        p = np.array([list(vector) for vector in model._df_user_vector['vector'].values])
        q = np.array([list(vector) for vector in model._df_movie_vector['vector'].values])

        restored_ratings = np.reshape(np.matmul(p, q.T), (-1,))
        expected_diff = np.absolute(ratings - restored_ratings)

        assert (expected_diff < config.FLOAT_EPSILN).all()

    def test_fit__non_full_matrix__restore_ratings(self, _non_full_matrix):
        user_movie_pair, ratings = _non_full_matrix

        model = RealValuedMatrixFactorization()
        model = model.fit(user_movie_pair, ratings, dim=3, epoch=300, lr=0.1, l1=0.0, l2=0.0)

        p = np.array([list(vector) for vector in model._df_user_vector['vector'].values])
        q = np.array([list(vector) for vector in model._df_movie_vector['vector'].values])

        argument_ratings = np.array([1, 2, 3, model._global_b])
        restored_ratings = np.reshape(np.matmul(p, q.T), (-1,))
        expected_diff = np.absolute(argument_ratings - restored_ratings)

        assert (expected_diff[:3] < config.FLOAT_EPSILN).all()
        assert expected_diff[3] < 1.0

    def test_predict__full_matrix__right_ratings(self, _non_full_matrix, _full_matrix):
        user_movie_pair, ratings = _non_full_matrix
        full_user_movie_pair, _ = _full_matrix

        model = RealValuedMatrixFactorization()
        model.fit(user_movie_pair, ratings, dim=3, epoch=300, lr=0.1, l1=0.0, l2=0.0)

        pred = model.predict(full_user_movie_pair)

        argument_ratings = np.array([1, 2, 3, model._global_b])
        expected_diff = np.absolute(argument_ratings - pred)

        assert (expected_diff[:3] < config.FLOAT_EPSILN).all()
        assert expected_diff[3] < 1.0

    def test_predict__include_nan_vector__right_ratings(self, _non_full_matrix, _full_matrix):
        df_user_vector = pd.DataFrame(
            [(0, np.array([1., 2., 3.])), (1, np.array([-1., -2., -3.]))],
            columns=['userId', 'vector'],
        )
        df_movie_vector = pd.DataFrame(
            [(100, np.array([0., 1., 3.])), (101, np.array([np.nan, np.nan, np.nan]))],
            columns=['movieId', 'vector'],
        )
        global_b = 123
        user_movie_pair = np.array([[0, 100], [0, 101], [1, 100]])

        model = RealValuedMatrixFactorization()
        model._df_user_vector = df_user_vector
        model._df_movie_vector = df_movie_vector
        model._global_b = global_b

        pred = model.predict(user_movie_pair)

        expected_ratings = np.array([11., 123., -11.])
        expected_diff = np.absolute(expected_ratings - pred)

        assert (expected_diff < config.FLOAT_EPSILN).all()

    def test_predict__include_unknown__right_ratings(self, _non_full_matrix, _full_matrix):
        df_user_vector = pd.DataFrame(
            [(0, np.array([1., 2., 3.])), (1, np.array([-1., -2., -3.]))],
            columns=['userId', 'vector'],
        )
        df_movie_vector = pd.DataFrame(
            [(100, np.array([0., 1., 3.]))],
            columns=['movieId', 'vector'],
        )
        global_b = 123
        user_movie_pair = np.array([[0, 100], [0, 101], [1, 100]])

        model = RealValuedMatrixFactorization()
        model._df_user_vector = df_user_vector
        model._df_movie_vector = df_movie_vector
        model._global_b = global_b

        pred = model.predict(user_movie_pair)

        expected_ratings = np.array([11., 123., -11.])
        expected_diff = np.absolute(expected_ratings - pred)

        assert (expected_diff < config.FLOAT_EPSILN).all()

    def test_save_and_load__do__restore(self, tmpdir):
        model = RealValuedMatrixFactorization()
        model._df_user_vector = pd.DataFrame(
            [[1, np.array([1, 2, 3])], [2, np.array([4, 5, 6])]],
            columns=['userId', 'vector'],
        )
        model._df_movie_vector = pd.DataFrame(
            [[100, np.array([-1, -2, -3])], [200, np.array([-4, -5, -6])]],
            columns=['movieId', 'vector'],
        )
        model._global_b = 0.5
        model.save(tmpdir)
        reloaded_model = RealValuedMatrixFactorization.load(tmpdir)

        pd.testing.assert_frame_equal(reloaded_model._df_user_vector, model._df_user_vector)
        pd.testing.assert_frame_equal(reloaded_model._df_movie_vector, model._df_movie_vector)
        assert reloaded_model._global_b == model._global_b
