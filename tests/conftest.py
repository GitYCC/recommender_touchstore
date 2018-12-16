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


@pytest.fixture
def user_movie_pair_1():
    return np.array([[1, 101],
                     [1, 103],
                     [2, 102],
                     [3, 102],
                     [3, 103]])


@pytest.fixture
def ratings_1():
    return np.array([5., 3., 2., 1., 4.])


@pytest.fixture
def user_movie_pair_2():
    return np.array([[1, 102],
                     [2, 101],
                     [2, 103],
                     [3, 101],
                     [4, 104]])


@pytest.fixture
def ratings_2():
    return np.array([1., 5., 3., 4., 2.])
