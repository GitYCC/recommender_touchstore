from abc import ABC, abstractmethod
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


class BaseModel(ABC):

    @abstractmethod
    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None, sample_weight=None):
        """Fit the model according to the given training data.

        Args:
            user_movie_pair ({array-like, sparse matrix}, shape (n_samples, 2)):
                Pair of userId and movieId, where n_samples is the number of samples.
            y (array-like, shape (n_samples,)):
                Target relative to user_movie_pair.
            user_feature (pandas.Dataframe, optional):
                Given more feature content about user.
            movie_feature (pandas.Dataframe, optional):
                Given more feature content about movie.
            sample_weight (array-like, shape (n_samples,), optional):
                Array of weights that are assigned to individual samples.
                If not provided, then each sample is given unit weight.

        Returns:
            self (object)

        """

    @abstractmethod
    def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
        """Predict target for samples in user_movie_pair.

        Args:
            user_movie_pair ({array-like, sparse matrix}, shape (n_samples, 2)):
                Pair of userId and movieId, where n_samples is the number of samples.
            user_feature (pandas.Dataframe, optional):
                Given more feature content about user.
            movie_feature (pandas.Dataframe, optional):
                Given more feature content about movie.

        Returns:
            y (array-like, shape (n_samples,)):
                Predicted target relative to user_movie_pair.

        """

    def _recommend_for_one_user(self, user_id, movies, user_feature, movie_feature, maxsize):
        user_movie_pair = np.array([[user_id, m] for m in movies], dtype='uint32')

        predicted = self.predict(user_movie_pair,
                                 user_feature=user_feature, movie_feature=movie_feature)

        n_samples = user_movie_pair.shape[0]
        predicted = predicted.reshape((n_samples, 1))
        table = np.hstack((user_movie_pair, predicted))
        df_table = pd.DataFrame(table, columns=['userId', 'movieId', 'predicted'])
        df_table = df_table.dropna()

        if maxsize is None:
            maxsize = len(movies)
        df_table['rank'] = df_table['predicted'].rank(ascending=False, method='first')
        df_table = df_table[df_table['rank'] <= maxsize]

        rec_items = np.full([1, maxsize], None, dtype='float64')
        rec_scores = np.full([1, maxsize], None, dtype='float64')
        for _, series in df_table.iterrows():
            movieId = series['movieId']
            score = series['predicted']
            rank = series['rank']

            index1_rec_items = int(rank - 1)
            rec_items[0, index1_rec_items] = movieId
            rec_scores[0, index1_rec_items] = score

        return (rec_items, rec_scores)

    def _recommend_for_one_movie(self, movie_id, users, user_feature, movie_feature, maxsize):
        user_movie_pair = np.array([[u, movie_id] for u in users], dtype='uint32')
        predicted = self.predict(user_movie_pair,
                                 user_feature=user_feature, movie_feature=movie_feature)

        n_samples = user_movie_pair.shape[0]
        predicted = predicted.reshape((n_samples, 1))
        table = np.hstack((user_movie_pair, predicted))
        df_table = pd.DataFrame(table, columns=['userId', 'movieId', 'predicted'])
        df_table = df_table.dropna()

        if maxsize is None:
            maxsize = len(users)
        df_table['rank'] = df_table['predicted'].rank(ascending=False, method='first')
        df_table = df_table[df_table['rank'] <= maxsize]

        rec_items = np.full([1, maxsize], None, dtype='float64')
        rec_scores = np.full([1, maxsize], None, dtype='float64')
        for _, series in df_table.iterrows():
            userId = series['userId']
            score = series['predicted']
            rank = series['rank']

            index1_rec_items = int(rank - 1)
            rec_items[0, index1_rec_items] = userId
            rec_scores[0, index1_rec_items] = score

        return (rec_items, rec_scores)

    def recommend(self, recommended_type, users, movies, user_feature=None, movie_feature=None,
                  maxsize=None):
        """Recommend items from type.

        Args:
            recommended_type (str): Recommended type, 'movie' or 'user'.
            users (list of int):
                User candidates present in userId.
            movies (list of int):
                Movie candidates present in movieId.
            user_feature (pandas.Dataframe, optional):
                Given more feature content about user.
            movie_feature (pandas.Dataframe, optional):
                Given more feature content about movie.

        Returns:
            recommended_items (array-like, shape (n_targets, n_recommended_items)):
                Recommended items in the order of predicted ranking.

        """
        rec_items, rec_scores = None, None

        if recommended_type == 'movie':
            print('recommend movies:')
            for i in tqdm(range(len(users))):
                sub_rec_items, sub_rec_scores = self._recommend_for_one_user(
                    users[i], movies, user_feature, movie_feature, maxsize)
                if rec_items is None and rec_scores is None:
                    rec_items, rec_scores = sub_rec_items, sub_rec_scores
                else:
                    rec_items = np.vstack((rec_items, sub_rec_items))
                    rec_scores = np.vstack((rec_scores, sub_rec_scores))

            assert rec_items.shape[0] == len(users)
            assert rec_scores.shape[0] == len(users)

        elif recommended_type == 'user':
            print('recommend users:')
            for i in tqdm(range(len(movies))):
                sub_rec_items, sub_rec_scores = self._recommend_for_one_movie(
                    movies[i], users, user_feature, movie_feature, maxsize)
                if rec_items is None and rec_scores is None:
                    rec_items, rec_scores = sub_rec_items, sub_rec_scores
                else:
                    rec_items = np.vstack((rec_items, sub_rec_items))
                    rec_scores = np.vstack((rec_scores, sub_rec_scores))

            assert rec_items.shape[0] == len(movies)
            assert rec_scores.shape[0] == len(movies)

        else:
            raise ValueError('wrong `recommended_type`')

        return (rec_items, rec_scores)

    @abstractmethod
    def _get_params(self):
        """Get parameters which determine this model.

        Returns:
            params (dict): parameters which determine this model.

        """

    @classmethod
    def load(cls, path_pickle):
        instance = object.__new__(cls)
        with open(path_pickle, 'rb') as input_file:
            params = pickle.load(input_file)
            for param_name, param_val in params.items():
                setattr(instance, param_name, param_val)
        return instance

    def save(self, path_pickle):
        params = self._get_params()
        with open(path_pickle, 'wb') as output_file:
            pickle.dump(params, output_file)
