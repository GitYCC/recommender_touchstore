from abc import ABC, abstractmethod


class BaseConverter(ABC):

    @abstractmethod
    def convert(self, datagroup, problem_type='rating_problem'):
        """Convert datagroup to the format for model fitting.

        Args:
            datagroup (process.Datagroup): group of data includes ratings, tags, movies and genome.
            problem_type (str): 'rating_problem' or 'like_problem'

        Returns:
            user_movie_pair ({array-like, sparse matrix}, shape (n_samples, 2)):
                Pair of userId and movieId, where n_samples is the number of samples.
            y (array-like, shape (n_samples,)):
                Target relative to user_movie_pair.
            user_feature (pandas.Dataframe or None):
                Given more feature content about user. If no user_feature, return None.
            movie_feature (pandas.Dataframe or None):
                Given more feature content about movie. If no movie_feature, return None.

        """


class NoContentConverter(BaseConverter):

    def convert(self, datagroup, problem_type='rating_problem'):
        if problem_type == 'rating_problem':
            df = datagroup.ratings
            kpi = 'rating'
        elif problem_type == 'like_problem':
            df = datagroup.likes
            kpi = 'like'

        user_movie_pair = df[['userId', 'movieId']].values
        y = df[kpi].values
        user_feature = None
        movie_feature = None
        return (user_movie_pair, y, user_feature, movie_feature)
