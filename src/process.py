import os
from datetime import datetime
from collections import namedtuple

import numpy as np
import pandas as pd

import config


def get_ratings():
    df = pd.read_csv(os.path.join(config.DIR_DATA, 'ratings_pub.csv'))
    return df


def _refine_movies(df):
    df['year'] = ((df['title'].str.extract(r'\((....)\) *$'))[0].astype('float32'))
    df = df.dropna()
    df['year'] = df['year'].astype('int32')
    df['title'] = (df['title'].str.extract(r'^(.*) \(....\) *$'))[0]
    df['genres'] = df['genres'].str.split('|')
    return df


def get_movies():
    df = pd.read_csv(os.path.join(config.DIR_DATA, 'movies_pub.csv'))
    df = _refine_movies(df)
    return df


def get_genome():
    df = pd.read_csv(os.path.join(config.DIR_DATA, 'genome_pub.csv'))
    return df


def get_tags():
    df = pd.read_csv(os.path.join(config.DIR_DATA, 'tags_pub.csv'))
    return df


def get_links():
    df = pd.read_csv(os.path.join(config.DIR_DATA, 'links_pub.csv'))
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
    df_ref_movies = pd.read_csv(os.path.join(config.DIR_DATA, 'ref_movies_q3.csv'))
    df_ref_movies = _refine_movies(df_ref_movies)
    df_ref_genome = pd.read_csv(os.path.join(config.DIR_DATA, 'ref_genome_q3.csv'))
    return df_ref_movies, df_ref_genome


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


Datagroup = namedtuple('Datagroup', ['ratings', 'tags', 'movies', 'genome'])


def split_datagroup(cut_year, datagroup):
    cut_dt = datetime(cut_year, 1, 1)
    df_ratings = datagroup.ratings
    df_tags = datagroup.tags
    df_movies = datagroup.movies
    df_genome = datagroup.genome

    df_ratings_before, df_ratings_after = \
        split_data_by_datetime(df_ratings, cut_dt)
    df_tags_before, df_tags_after = \
        split_data_by_datetime(df_tags, cut_dt)
    df_movies_before, df_movies_after = \
        split_data_by_year(df_movies, cut_year)
    df_genome_before = select_data_by_movie_group(df_genome, df_movies_before.movieId)
    df_genome_after = select_data_by_movie_group(df_genome, df_movies_after.movieId)

    datagroup_before = Datagroup(ratings=df_ratings_before,
                                 tags=df_tags_before,
                                 movies=df_movies_before,
                                 genome=df_genome_before)
    datagroup_after = Datagroup(ratings=df_ratings_after,
                                tags=df_tags_after,
                                movies=df_movies_after,
                                genome=df_genome_after)
    return (datagroup_before, datagroup_after)


def save_datagroup(folder, datagroup, postfix):
    paths = list()
    for key in ['ratings', 'tags', 'movies', 'genome']:
        df = getattr(datagroup, key)
        path = os.path.join(folder, key + '_' + postfix + '.csv')
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def load_datagroup(folder, postfix):
    dfs = dict()
    for key in ['ratings', 'tags', 'movies', 'genome']:
        dfs[key] = pd.read_csv(os.path.join(folder, key + '_' + postfix + '.csv'))

    # special handle on movies' genres
    dfs['movies']['genres'] = dfs['movies']['genres'].map(lambda x: eval(x))

    datagroup = Datagroup(**dfs)
    return datagroup
