from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):

    @abstractmethod
    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None, sample_weight=None):
        """Fit the model according to the given training data.

        Args:
            user_movie_pair ({array-like, sparse matrix}, shape (n_samples, 2)):
                Pair of userId and movieId, where n_samples is the number of samples.
            y (array-like, shape (n_samples,)):
                Target relative to user_movie_pair.
            user_feature (array-like, shape (n_userId, n_user_feature), optional):
                Given more feature content about user.
            movie_feature (array-like, shape (n_movieId, n_movie_feature), optional):
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
            user_feature (array-like, shape (n_userId, n_user_feature), optional):
                Given more feature content about user.
            movie_feature (array-like, shape (n_movieId, n_movie_feature), optional):
                Given more feature content about movie.

        Returns:
            y (array-like, shape (n_samples,)):
                Predicted target relative to user_movie_pair.

        """

    def recommend(self, recommended_type, users, movies, user_feature=None, movie_feature=None):
        """Recommend items from type.

        Args:
            recommended_type (str): Recommended type, 'movie' or 'user'.
            users (list of int):
                User candidates present in userId.
            movies (list of int):
                Movie candidates present in movieId.
            user_feature (array-like, shape (n_userId, n_user_feature), optional):
                Given more feature content about user.
            movie_feature (array-like, shape (n_movieId, n_movie_feature), optional):
                Given more feature content about movie.

        Returns:
            recommended_items (array-like, shape (n_targets, n_recommended_items)):
                Recommended items in the order of predicted ranking.

        """
        user_movie_pair = np.array([[u, m] for m in movies for u in users], dtype='uint32')
        predicted = self.predict(user_movie_pair,
                                 user_feature=user_feature, movie_feature=movie_feature)
        n_samples = user_movie_pair.shape[0]
        predicted = predicted.reshape((n_samples, 1))
        table = np.hstack((predicted, user_movie_pair))
        df_table = pd.DataFrame(table, columns=['predicted', 'userId', 'movieId'])

        if recommended_type == 'movie':
            df_table['rank'] = \
                df_table.groupby('userId')['predicted'].rank(ascending=False, method='min')
            result = np.full([len(users), len(movies)], 0, dtype='uint32')
            userId_index_map = dict()
            for index, series in df_table.iterrows():
                userId = series['userId']
                movieId = series['movieId']
                rank = series['rank']
                if userId not in userId_index_map:
                    userId_index_map[userId] = users.index(userId)
                index0_result = int(userId_index_map[userId])
                index1_result = int(rank - 1)
                while result[index0_result, index1_result] != 0:
                    index1_result += 1
                result[index0_result, index1_result] = movieId

        elif recommended_type == 'user':
            df_table['rank'] = \
                df_table.groupby('movieId')['predicted'].rank(ascending=False, method='min')
            result = np.full([len(movies), len(users)], 0, dtype='uint32')
            movieId_index_map = dict()
            for index, series in df_table.iterrows():
                userId = series['userId']
                movieId = series['movieId']
                rank = series['rank']
                if movieId not in movieId_index_map:
                    movieId_index_map[movieId] = movies.index(movieId)
                index0_result = int(movieId_index_map[movieId])
                index1_result = int(rank - 1)
                while result[index0_result, index1_result] != 0:
                    index1_result += 1
                result[index0_result, index1_result] = userId

        else:
            raise ValueError('wrong `recommended_type`')

        return result
