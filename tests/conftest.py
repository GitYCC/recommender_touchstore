import sys

sys.path.insert(0, 'src')

import pytest
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
