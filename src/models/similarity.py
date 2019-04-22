import logging

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .model import BaseModel

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ItemCosineSimilarity(BaseModel):
    def __init__(self):
        self._df_ratings = None
        self._df_avg_y = None

    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None,
            similarity_theshold=0):
        y = np.reshape(y, (y.shape[0], 1))
        content = np.hstack((user_movie_pair, y))
        df = pd.DataFrame(
            content,
            columns=['userId', 'movieId', 'y'],
        )
        df.userId = df.userId.astype('int32')
        df.movieId = df.movieId.astype('int32')

        # indexize
        logger.info('indexize')

        df_user_id = df[['userId']].drop_duplicates()
        df_user_id['user_index'] = np.array(list(range(df_user_id.shape[0])))

        df_movie_id = df[['movieId']].drop_duplicates()
        df_movie_id['movie_index'] = np.array(list(range(df_movie_id.shape[0])))

        df = pd.merge(df, df_user_id, on='userId').drop(columns=['userId'])
        df = pd.merge(df, df_movie_id, on='movieId').drop(columns=['movieId'])
        df = df[['user_index', 'movie_index', 'y']]

        # mean centering
        logger.info('mean centering')

        df_avg_y = df[['movie_index', 'y']] \
            .groupby('movie_index', as_index=False, sort=False).mean() \
            .rename(columns={'y': 'avg_y'})

        df = pd.merge(df, df_avg_y, on='movie_index')
        df['mean_centering_y'] = df['y'] - df['avg_y']
        df = df[['user_index', 'movie_index', 'mean_centering_y']]

        # calculate item similarity
        logger.info('calculate item similarity step 1: cosine')

        sparse_mat = coo_matrix((df.mean_centering_y, (df.movie_index, df.user_index)))
        similarities_sparse = cosine_similarity(sparse_mat, dense_output=False).todok()

        logger.info('calculate item similarity step 2: filter by threshold {}' \
                    .format(similarity_theshold))

        similarity_content = []
        for (movie_index, reference), rating in tqdm(similarities_sparse.items()):
            if abs(rating) > similarity_theshold:
                similarity_content.append((movie_index, reference, rating))
        df_similarity = pd.DataFrame(
            similarity_content,
            columns=['movie_index', 'reference_movie_index', 'sim'])

        # calculate relative rating
        logger.info('calculate relative rating')

        df = pd.merge(df, df_similarity, how='right', on='movie_index')
        df = df[['user_index', 'reference_movie_index', 'mean_centering_y', 'sim']] \
            .rename(columns={'reference_movie_index': 'movie_index'})
        df['mcy_mul_sim'] = df['mean_centering_y'] * df['sim']
        df['abs_sim'] = np.absolute(df['sim'])
        df = df.groupby(['user_index', 'movie_index'], as_index=False, sort=False).sum()
        df['relative_rating'] = df['mcy_mul_sim'] / df['abs_sim']
        df = df[['user_index', 'movie_index', 'relative_rating']]

        # calculate rating
        logger.info('calculate rating')

        df = pd.merge(df, df_avg_y, on='movie_index')
        df['rating'] = df['relative_rating'] + df['avg_y']
        df = df[['user_index', 'movie_index', 'rating']]

        # back by indexer
        logger.info('back by indexer')

        df = pd.merge(df, df_user_id, on='user_index')
        df = pd.merge(df, df_movie_id, on='movie_index')
        df = df[['userId', 'movieId', 'rating']]

        df_avg_y = pd.merge(df_avg_y, df_movie_id, on='movie_index')
        df_avg_y = df_avg_y[['movieId', 'avg_y']]

        self._df_ratings = df
        self._df_avg_y = df_avg_y
        return self

    def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
        df = pd.DataFrame(
            user_movie_pair,
            columns=['userId', 'movieId'],
        )
        df = pd.merge(df, self._df_ratings, how='left', on=['userId', 'movieId'])
        null_filter = df.rating.isnull()
        df.loc[null_filter, 'rating'] = \
            pd.merge(df.loc[null_filter, :], self._df_avg_y, how='left', on='movieId')['avg_y'] \
            .values  # important necessary: convert to ndarray
        return df['rating'].values

    @classmethod
    def load(cls, local_dir):
        instance = ItemCosineSimilarity()
        instance._df_ratings = pd.read_pickle(str(local_dir / 'df_ratings.pkl'))
        instance._df_avg_y = pd.read_pickle(str(local_dir / 'df_avg_y.pkl'))
        return instance

    def save(self, local_dir):
        self._df_ratings.to_pickle(str(local_dir / 'df_ratings.pkl'))
        self._df_avg_y.to_pickle(str(local_dir / 'df_avg_y.pkl'))
