from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm


class BaseModel(ABC):

    @abstractmethod
    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None, **model_params):
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

    def _recommend_for_users(self, users, movies, user_feature, movie_feature, maxsize):
        user_movie_pair = np.array([[u, m] for m in movies for u in users], dtype='uint32')
        user_index_mapping = {u: i for i, u in enumerate(users)}

        predicted = self.predict(user_movie_pair,
                                 user_feature=user_feature, movie_feature=movie_feature)

        n_samples = user_movie_pair.shape[0]
        predicted = predicted.reshape((n_samples, 1))
        table = np.hstack((user_movie_pair, predicted))
        df_table = pd.DataFrame(table, columns=['userId', 'movieId', 'predicted'])
        df_table = df_table.dropna()

        if maxsize is None:
            maxsize = len(movies)
        df_table['rank'] = df_table.groupby('userId', as_index=False, sort=False)['predicted'] \
                                   .rank(ascending=False, method='first')
        df_table = df_table[df_table['rank'] <= maxsize]

        rec_items = np.full([len(users), maxsize], None, dtype='float64')
        rec_scores = np.full([len(users), maxsize], None, dtype='float64')
        for _, series in df_table.iterrows():
            userId = series['userId']
            movieId = series['movieId']
            score = series['predicted']
            rank = series['rank']

            index0 = int(user_index_mapping[userId])
            index1_rec_items = int(rank - 1)
            rec_items[index0, index1_rec_items] = movieId
            rec_scores[index0, index1_rec_items] = score

        return (rec_items, rec_scores)

    def _recommend_for_movies(self, movies, users, user_feature, movie_feature, maxsize):
        user_movie_pair = np.array([[u, m] for u in users for m in movies], dtype='uint32')
        movie_index_mapping = {m: i for i, m in enumerate(movies)}

        predicted = self.predict(user_movie_pair,
                                 user_feature=user_feature, movie_feature=movie_feature)

        n_samples = user_movie_pair.shape[0]
        predicted = predicted.reshape((n_samples, 1))
        table = np.hstack((user_movie_pair, predicted))
        df_table = pd.DataFrame(table, columns=['userId', 'movieId', 'predicted'])
        df_table = df_table.dropna()

        if maxsize is None:
            maxsize = len(users)
        df_table['rank'] = df_table.groupby('movieId', as_index=False, sort=False)['predicted'] \
                                   .rank(ascending=False, method='first')
        df_table = df_table[df_table['rank'] <= maxsize]

        rec_items = np.full([len(movies), maxsize], None, dtype='float64')
        rec_scores = np.full([len(movies), maxsize], None, dtype='float64')
        for _, series in df_table.iterrows():
            movieId = series['movieId']
            userId = series['userId']
            score = series['predicted']
            rank = series['rank']

            index0 = int(movie_index_mapping[movieId])
            index1_rec_items = int(rank - 1)
            rec_items[index0, index1_rec_items] = userId
            rec_scores[index0, index1_rec_items] = score

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
            maxsize (int):
                Count of recommendation items

        Returns:
            recommended_items (array-like, shape (n_targets, n_recommended_items)):
                Recommended items in the order of predicted ranking.

        """
        rec_items, rec_scores = None, None

        batch_num = 100

        if recommended_type == 'movie':
            print('recommend movies:')
            user_num = len(users)
            with tqdm(total=user_num) as pbar:
                for i in range(1+user_num//batch_num):
                    start, end = i * batch_num, min((i+1) * batch_num, user_num)
                    sub_rec_items, sub_rec_scores = self._recommend_for_users(
                        users[start:end], movies, user_feature, movie_feature, maxsize)
                    if rec_items is None and rec_scores is None:
                        rec_items, rec_scores = sub_rec_items, sub_rec_scores
                    else:
                        rec_items = np.vstack((rec_items, sub_rec_items))
                        rec_scores = np.vstack((rec_scores, sub_rec_scores))
                    pbar.update(end - start)

            assert rec_items.shape[0] == len(users)
            assert rec_scores.shape[0] == len(users)

        elif recommended_type == 'user':
            print('recommend users:')
            movie_num = len(movies)
            with tqdm(total=movie_num) as pbar:
                for i in range(1+movie_num//batch_num):
                    start, end = i * batch_num, min((i+1) * batch_num, movie_num)
                    sub_rec_items, sub_rec_scores = self._recommend_for_movies(
                        movies[start:end], users, user_feature, movie_feature, maxsize)
                    if rec_items is None and rec_scores is None:
                        rec_items, rec_scores = sub_rec_items, sub_rec_scores
                    else:
                        rec_items = np.vstack((rec_items, sub_rec_items))
                        rec_scores = np.vstack((rec_scores, sub_rec_scores))
                    pbar.update(end - start)

            assert rec_items.shape[0] == len(movies)
            assert rec_scores.shape[0] == len(movies)

        else:
            raise ValueError('wrong `recommended_type`')

        return (rec_items, rec_scores)

    @classmethod
    @abstractmethod
    def load(cls, local_dir):
        """Load model.

        Args:
            local_dir (pathlib.Path): Directory of loading.

        """

    @abstractmethod
    def save(self, local_dir):
        """Save model.

        Args:
            local_dir (pathlib.Path): Directory of saving.

        """
