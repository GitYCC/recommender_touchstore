import sys

sys.path.insert(0, 'src')

import pytest
import numpy as np
import pandas as pd

from process import Datagroup


@pytest.fixture
def datagroup():
    return Datagroup(ratings=pd.DataFrame({'userId': [1, 2, 3],
                                           'movieId': [101, 102, 103],
                                           'rating': [5, 4, 3],
                                           'timestamp': [1112486027, 1112484727, 1094785740]}),
                     tags=pd.DataFrame({'userId': [1, 2, 2],
                                        'movieId': [101, 102, 103],
                                        'tag': ['Mark Waters', 'New Zealand', 'surreal']}),
                     movies=pd.DataFrame({'movieId': [101, 102, 103],
                                          'title': ['A', 'B', 'C'],
                                          'genres': [['Drama', 'Fantasy', 'Romance'],
                                                     ['Drama', ],
                                                     ['Fantasy', 'Romance']],
                                          'year': [2010, 2014, 2009]}),
                     genome=pd.DataFrame({'movieId': [101, 102, 103],
                                          'tagId': [1001, 1001, 1001],
                                          'relevance': [0.5, 0.4, 0.9],
                                          'tag': ['yc', 'yc', 'yc']}),
                     )


# user_movie_pair_1 and ratings_1
# users/movies  100  101  102  103  104  105
# 0              4    1    -    1    -    -
# 1              2    -    2    -    4    -
# 2              3    -    3    3    -    -
# 3              -    -    4    -    5    5
@pytest.fixture
def user_movie_pair_1():
    return np.array([[0, 100], [0, 101], [0, 103],
                     [1, 100], [1, 102], [1, 104],
                     [2, 100], [2, 102], [2, 103],
                     [3, 102], [3, 104], [3, 105]])


@pytest.fixture
def ratings_1():
    return np.array([4., 1., 1.,
                     2., 2., 4.,
                     3., 3., 3.,
                     4., 5., 5.])


# user_movie_pair_1_opp
# users/movies  100  101  102  103  104  105
# 0              -    -    o    -    o    o
# 1              -    o    -    o    -    o
# 2              -    o    -    -    o    o
# 3              o    o    -    o    -    -
@pytest.fixture
def user_movie_pair_1_opp():
    return np.array([[0, 102], [0, 104], [0, 105],
                     [1, 101], [1, 103], [1, 105],
                     [2, 101], [2, 104], [2, 105],
                     [3, 100], [3, 101], [3, 103]])


# user_movie_pair_2 and ratings_2
# users/movies  100  101  102  103  104  105  106
# 0              -    -    3    -    -    -    4
# 1              2    -    -    -    -    2    -
# 2              -    5    -    -    -    -    3
# 3              3    -    -    -    -    -    -
# 4              -    4    -    1    1    3    -
@pytest.fixture
def user_movie_pair_2():
    return np.array([[0, 102], [0, 106],
                     [1, 100], [1, 105],
                     [2, 101], [2, 106],
                     [3, 100],
                     [4, 101], [4, 103], [4, 104], [4, 105]])


@pytest.fixture
def ratings_2():
    return np.array([3., 4.,
                     2., 2.,
                     5., 3.,
                     3.,
                     4., 1., 1., 3.])
