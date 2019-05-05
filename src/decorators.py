import logging
import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from process import Datagroup

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDecorator(ABC):

    @abstractmethod
    def decorate(self, datagroup, problem_type='like_problem', **decorate_params):
        """Decorate datagroup.

        Args:
            datagroup (process.Datagroup): group of data includes ratings, likes, movie_feature
                                           and user_feature.
            problem_type (str): 'rating_problem' or 'like_problem'

        Returns:
            decorated_datagroup (process.Datagroup): decorated group of data.

        """


class NegativeDataDecorator(BaseDecorator):

    def decorate(self, datagroup, problem_type='like_problem',
                 negative_data_ratio=1.0, negative_data_value=0):
        if problem_type == 'rating_problem':
            df = datagroup.ratings
            kpi = 'rating'
            logger.error('Applying NegativeDataDecorator is not suitable in rating problem')
        elif problem_type == 'like_problem':
            df = datagroup.likes
            kpi = 'like'

        positive_data_count = df.shape[0]
        negative_data_count = int(positive_data_count * negative_data_ratio)

        um_pair = df[['userId', 'movieId']].values
        users = np.unique(um_pair[:, 0]).tolist()
        movies = np.unique(um_pair[:, 1]).tolist()
        collect_data = set((u, m) for u, m in um_pair.tolist())

        i = 0
        negative_data = []
        while i < negative_data_count:
            user = random.choice(users)
            movie = random.choice(movies)
            pair = (user, movie)
            if pair not in collect_data:
                negative_data.append(pair)
                collect_data.add(pair)
                i += 1

        df_negative = pd.DataFrame(
            [(u, m, negative_data_value) for u, m in negative_data],
            columns=['userId', 'movieId', kpi],
        )
        df_full = pd.concat([df, df_negative], axis=0, sort=False)
        df_full = df_full.sample(frac=1).reset_index(drop=True)

        if problem_type == 'rating_problem':
            decorated_datagroup = Datagroup(
                ratings=df_full,
                likes=None,
                movie_feature=datagroup.movie_feature,
                user_feature=datagroup.user_feature,
            )
        elif problem_type == 'like_problem':
            decorated_datagroup = Datagroup(
                ratings=None,
                likes=df_full,
                movie_feature=datagroup.movie_feature,
                user_feature=datagroup.user_feature,
            )
        return decorated_datagroup
