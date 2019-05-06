import numpy as np
import pandas as pd

from .model import BaseModel


class PopularityModel(BaseModel):
    def __init__(self):
        self._rating_avg = dict()

    def fit(self, user_movie_pair, y, user_feature=None, movie_feature=None,
            valid_user_movie_pair=None, valid_y=None,
            valid_user_feature=None, valid_movie_feature=None):
        y = np.reshape(y, (y.shape[0], 1))
        content = np.hstack((user_movie_pair, y))
        df = pd.DataFrame(
            content,
            columns=['userId', 'movieId', 'y'],
        )
        df.userId = df.userId.astype('int32')
        df.movieId = df.movieId.astype('int32')

        df_rating_avg = df.drop(columns=['userId']) \
                          .groupby('movieId', as_index=False, sort=False).mean()

        self._df_rating_avg = df_rating_avg

        return self

    def predict(self, user_movie_pair, user_feature=None, movie_feature=None):
        df = pd.DataFrame(
            user_movie_pair[:, 1],
            columns=['movieId']
        )
        df = pd.merge(df, self._df_rating_avg, on='movieId', how='left')
        return df['y'].values

    @classmethod
    def load(cls, local_dir):
        instance = PopularityModel()
        instance._df_rating_avg = pd.read_pickle(str(local_dir / 'df_rating_avg.pkl'))
        return instance

    def save(self, local_dir):
        self._df_rating_avg.to_pickle(str(local_dir / 'df_rating_avg.pkl'))
