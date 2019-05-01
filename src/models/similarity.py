import logging

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

from .model import BaseModel

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ItemCosineSimilarity(BaseModel):
    def __init__(self):
        self._df_ratings = None
        self._df_center_y = None

    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None,
            valid_user_movie_pair=None, valid_y=None,
            valid_user_feature=None, valid_movie_feature=None,
            similarity_theshold=0, use_mean_centering=True):
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
        if use_mean_centering:
            logger.info('mean centering')

            df_center_y = df[['movie_index', 'y']] \
                .groupby('movie_index', as_index=False, sort=False).mean() \
                .rename(columns={'y': 'center_y'})

            df = pd.merge(df, df_center_y, on='movie_index')
            df['centered_y'] = df['y'] - df['center_y']
            df = df[['user_index', 'movie_index', 'centered_y']]
        else:
            df_center_y = df_movie_id[['movie_index']]
            df_center_y['center_y'] = 0.0
            df = df.rename(columns={'y': 'centered_y'})
            df = df[['user_index', 'movie_index', 'centered_y']]

        # calculate item similarity
        logger.info('calculate item similarity step 1: cosine')

        sparse_mat = coo_matrix((df.centered_y, (df.movie_index, df.user_index)))
        similarities_sparse = cosine_similarity(sparse_mat, dense_output=False)

        logger.info('calculate item similarity step 2: filter by threshold {}'
                    .format(similarity_theshold))

        mask = (np.absolute(similarities_sparse) > similarity_theshold)
        similarities_sparse = similarities_sparse.multiply(mask)
        similarities_sparse = similarities_sparse.tocoo()

        df_similarity = pd.DataFrame(
            dict(
                movie_index=similarities_sparse.row,
                reference_movie_index=similarities_sparse.col,
                sim=similarities_sparse.data,
            ))

        # calculate relative rating
        logger.info('calculate relative rating')

        df = pd.merge(df, df_similarity, how='right', on='movie_index')
        df = df[['user_index', 'reference_movie_index', 'centered_y', 'sim']] \
            .rename(columns={'reference_movie_index': 'movie_index'})
        df['cy_mul_sim'] = df['centered_y'] * df['sim']
        df['abs_sim'] = np.absolute(df['sim'])
        df = df.groupby(['user_index', 'movie_index'], as_index=False, sort=False).sum()
        df['relative_rating'] = df['cy_mul_sim'] / df['abs_sim']
        df = df[['user_index', 'movie_index', 'relative_rating']]

        # calculate rating
        logger.info('calculate rating')

        df = pd.merge(df, df_center_y, on='movie_index')
        df['rating'] = df['relative_rating'] + df['center_y']
        df = df[['user_index', 'movie_index', 'rating']]

        # back by indexer
        logger.info('back by indexer')

        df = pd.merge(df, df_user_id, on='user_index')
        df = pd.merge(df, df_movie_id, on='movie_index')
        df = df[['userId', 'movieId', 'rating']]

        df_center_y = pd.merge(df_center_y, df_movie_id, on='movie_index')
        df_center_y = df_center_y[['movieId', 'center_y']]

        self._df_ratings = df
        self._df_center_y = df_center_y
        return self

    def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
        df = pd.DataFrame(
            user_movie_pair,
            columns=['userId', 'movieId'],
        )
        df = pd.merge(df, self._df_ratings, how='left', on=['userId', 'movieId'])
        null_filter = df.rating.isnull()
        df.loc[null_filter, 'rating'] = \
            pd.merge(
                df.loc[null_filter, :],
                self._df_center_y,
                how='left',
                on='movieId',
            )['center_y'].values  # important necessary: convert to ndarray
        return df['rating'].values

    @classmethod
    def load(cls, local_dir):
        instance = ItemCosineSimilarity()
        instance._df_ratings = pd.read_pickle(str(local_dir / 'df_ratings.pkl'))
        instance._df_center_y = pd.read_pickle(str(local_dir / 'df_center_y.pkl'))
        return instance

    def save(self, local_dir):
        self._df_ratings.to_pickle(str(local_dir / 'df_ratings.pkl'))
        self._df_center_y.to_pickle(str(local_dir / 'df_center_y.pkl'))
