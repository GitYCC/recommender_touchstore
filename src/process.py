import os
from datetime import datetime
from collections import namedtuple

import numpy as np
import pandas as pd

import config

Datagroup = namedtuple('Datagroup', ['ratings', 'likes', 'movie_feature', 'user_feature'])


def get_ratings():
    df = pd.read_pickle(os.path.join(config.DIR_DATA, 'ratings_pub.pkl'))
    return df


def get_likes():
    df = pd.read_pickle(os.path.join(config.DIR_DATA, 'likes_pub.pkl'))
    return df


def get_movie_feature():
    df = pd.read_pickle(os.path.join(config.DIR_DATA, 'movie_feature_pub.pkl'))
    return df


def get_question1():
    df = pd.read_csv(os.path.join(config.DIR_DATA, 'test_q1.csv'))
    df['pred_rating'] = np.nan
    return df


def get_answer1():
    ans = list()
    with open(os.path.join(config.DIR_PRIVATE, 'ans_q1.txt')) as fr:
        for line in fr.readlines():
            ans.append(float(line.strip()))
    ans = np.array(ans)
    return ans


def get_question2():
    users = dict()
    with open(os.path.join(config.DIR_DATA, 'test_q2.txt')) as fr:
        users['userId'] = [int(line.strip()) for line in fr.readlines()]
    df = pd.DataFrame(users, columns=['userId'])
    df['pred_movies'] = np.nan
    return df


def get_answer2():
    ans = list()
    with open(os.path.join(config.DIR_PRIVATE, 'ans_q2.txt')) as fr:
        for line in fr.readlines():
            list_ = [int(x) for x in line.strip().split(',')]
            ans.append(list_)
    return ans


def get_question3():
    movies = dict()
    with open(os.path.join(config.DIR_DATA, 'test_q3.txt')) as fr:
        movies['movieId'] = [int(line.strip()) for line in fr.readlines()]
    df = pd.DataFrame(movies, columns=['movieId'])
    df['pred_users'] = np.nan
    return df


def get_question3_ref():
    df_ref_movie_feature = pd.read_pickle(os.path.join(config.DIR_DATA, 'ref_movie_feature.pkl'))
    return df_ref_movie_feature


def get_answer3():
    ans = list()
    with open(os.path.join(config.DIR_PRIVATE, 'ans_q3.txt')) as fr:
        for line in fr.readlines():
            list_ = [int(x) for x in line.strip().split(',')]
            ans.append(list_)
    return ans


def _dt2ts(dt):
    timestamp = int((dt - datetime(1970, 1, 1)).total_seconds())
    return timestamp


def split_data_by_datetime(df_including_timestamp, cut_dt):
    cut_timestamp = _dt2ts(cut_dt)
    df_before = (df_including_timestamp[df_including_timestamp.timestamp < cut_timestamp]
                 .reset_index(drop=True))
    df_after = (df_including_timestamp[df_including_timestamp.timestamp >= cut_timestamp]
                .reset_index(drop=True))
    return (df_before, df_after)


def split_data_by_year(df_including_year, cut_year):
    df_before = (df_including_year[df_including_year.year < cut_year]
                 .reset_index(drop=True))
    df_after = (df_including_year[df_including_year.year >= cut_year]
                .reset_index(drop=True))
    return (df_before, df_after)


def select_data_by_movie_group(df_including_movieId, movie_group):  # noqa: N803
    return (df_including_movieId[df_including_movieId.movieId.isin(movie_group)]
            .reset_index(drop=True))


def split_datagroup(cut_year, datagroup):
    cut_dt = datetime(cut_year, 1, 1)
    df_ratings = datagroup.ratings
    df_likes = datagroup.likes
    df_movie_feature = datagroup.movie_feature

    df_ratings_before, df_ratings_after = \
        split_data_by_datetime(df_ratings, cut_dt)
    df_likes_before, df_likes_after = \
        split_data_by_datetime(df_likes, cut_dt)
    df_movie_feature_before, df_movie_feature_after = \
        split_data_by_year(df_movie_feature, cut_year)

    datagroup_before = Datagroup(ratings=df_ratings_before,
                                 likes=df_likes_before,
                                 movie_feature=df_movie_feature_before,
                                 user_feature=None)
    datagroup_after = Datagroup(ratings=df_ratings_after,
                                likes=df_likes_after,
                                movie_feature=df_movie_feature_after,
                                user_feature=None)
    return (datagroup_before, datagroup_after)


def save_datagroup(folder, datagroup, postfix):
    paths = list()
    for key in ['ratings', 'likes', 'movie_feature', 'user_feature']:
        df = getattr(datagroup, key)
        if df is not None:
            path = os.path.join(folder, key + '_' + postfix + '.pkl')
            df.to_pickle(path)
            paths.append(path)
    return paths


def load_datagroup(folder, postfix):
    dfs = dict()
    for key in ['ratings', 'likes', 'movie_feature', 'user_feature']:
        path = os.path.join(folder, key + '_' + postfix + '.pkl')
        dfs[key] = pd.read_pickle(path) if os.path.exists(path) else None

    datagroup = Datagroup(**dfs)
    return datagroup
