import os

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

WORKSPACE = os.path.dirname(__file__)
DIR_PUBLIC_DATA = os.path.join(WORKSPACE, '..', 'data')


def _get_old_movies():
    df_movies_public = pd.read_csv(os.path.join(DIR_PUBLIC_DATA, 'movies_pub.csv'))
    return df_movies_public.movieId


def _get_new_movies():
    df_movies_private = pd.read_csv(os.path.join(WORKSPACE, 'movies_prv.csv'))
    return df_movies_private.movieId


def prepare_question1():
    """Prepare Question 1.

    Rating Problem: Design a system to predict an unknown rating when given `userId` and `movieId`.

    """
    df_ratings_public = pd.read_csv(os.path.join(DIR_PUBLIC_DATA, 'ratings_pub.csv'))
    df_ratings_private = pd.read_csv(os.path.join(WORKSPACE, 'ratings_prv.csv'))

    old_users = df_ratings_public['userId'].unique()
    old_movies = _get_old_movies()
    common = df_ratings_private.loc[:, ('userId', 'movieId')].merge(
        df_ratings_public.loc[:, ('userId', 'movieId')], on=('userId', 'movieId'))

    selected = (
        df_ratings_private.userId.isin(old_users) &
        df_ratings_private.movieId.isin(old_movies) &
        ((~df_ratings_private.userId.isin(common.userId)) &
         (~df_ratings_private.movieId.isin(common.movieId)))
    )
    df_q1 = df_ratings_private[selected]

    path_test = os.path.join(DIR_PUBLIC_DATA, 'test_q1.csv')
    path_answer = os.path.join(WORKSPACE, 'ans_q1.txt')

    df_q1.loc[:, ('userId', 'movieId')].to_csv(path_test, index=None)
    df_q1.loc[:, 'rating'].to_csv(path_answer, index=None, header=False)


def prepare_question2():
    """Prepare Question 2.

    Ranking Problem: We defined that the rating > 3.0 as a favorite movie. Design a system to
    recommend a top-10 favorite movies for a person, a `movieId` list which `userId` did not
    see before (provide `userId` at `./src/data/test_q2.txt`).

    """
    threshold_movies = 5

    df_likes_public = pd.read_csv(os.path.join(DIR_PUBLIC_DATA, 'likes_pub.csv'))
    df_likes_private = pd.read_csv(os.path.join(WORKSPACE, 'likes_prv.csv'))

    old_users = df_likes_public['userId'].unique()
    old_movies = _get_old_movies()
    common = df_likes_private.loc[:, ('userId', 'movieId')].merge(
        df_likes_public.loc[:, ('userId', 'movieId')], on=('userId', 'movieId'))

    selected = (
        df_likes_private.userId.isin(old_users) &
        df_likes_private.movieId.isin(old_movies) &
        ((~df_likes_private.userId.isin(common.userId)) &
         (~df_likes_private.movieId.isin(common.movieId)))
    )
    df = df_likes_private[selected]
    dict_q2 = df.groupby('userId')['movieId'].apply(list).dropna().to_dict()

    path_test = os.path.join(DIR_PUBLIC_DATA, 'test_q2.txt')
    path_answer = os.path.join(WORKSPACE, 'ans_q2.txt')

    fw_test = open(path_test, 'w')
    fw_answer = open(path_answer, 'w')

    for u, movies in dict_q2.items():
        if len(movies) <= threshold_movies:
            continue
        fw_test.write('{}\n'.format(int(u)))
        fw_answer.write('{}\n'.format(','.join(map(str, movies))))

    fw_test.close()
    fw_answer.close()


def prepare_question3():
    """Prepare Question 3.

    Content-based Problem: We defined that the rating > 3.0 as a favorite movie.
    Design a system to recommend a top-10 `userId` they may like a new movie
    (at `./src/data/test_q3.txt`). We will give you some information of that new movie
    (at `./src/data/ref_movies_q3.csv` and `./src/data/ref_genome_q3.csv`).

    """
    threshold_users = 5

    df_likes_public = pd.read_csv(os.path.join(DIR_PUBLIC_DATA, 'likes_pub.csv'))
    df_likes_private = pd.read_csv(os.path.join(WORKSPACE, 'likes_prv.csv'))

    old_users = df_likes_public['userId'].unique()
    new_movies = _get_new_movies()
    common = df_likes_private.loc[:, ('userId', 'movieId')].merge(
        df_likes_public.loc[:, ('userId', 'movieId')], on=('userId', 'movieId'))

    selected = (
        df_likes_private.userId.isin(old_users) &
        df_likes_private.movieId.isin(new_movies) &
        ((~df_likes_private.userId.isin(common.userId)) &
         (~df_likes_private.movieId.isin(common.movieId)))
    )
    df = df_likes_private[selected]
    dict_q3 = df.groupby('movieId')['userId'].apply(list).dropna().to_dict()

    path_test = os.path.join(DIR_PUBLIC_DATA, 'test_q3.txt')
    path_answer = os.path.join(WORKSPACE, 'ans_q3.txt')

    fw_test = open(path_test, 'w')
    fw_answer = open(path_answer, 'w')

    selected_movies = []
    for m, users in dict_q3.items():
        if len(users) <= threshold_users:
            continue
        selected_movies.append(m)
        fw_test.write('{}\n'.format(int(m)))
        fw_answer.write('{}\n'.format(','.join(map(str, users))))

    fw_test.close()
    fw_answer.close()

    df_movies_private = pd.read_csv(os.path.join(WORKSPACE, 'movies_prv.csv'))
    df_genome_private = pd.read_csv(os.path.join(WORKSPACE, 'genome_prv.csv'))

    df_movies_ref = df_movies_private[df_movies_private.movieId.isin(selected_movies)]
    df_genome_ref = df_genome_private[df_genome_private.movieId.isin(selected_movies)]

    path_movies_ref = os.path.join(DIR_PUBLIC_DATA, 'ref_movies_q3.csv')
    path_genome_ref = os.path.join(DIR_PUBLIC_DATA, 'ref_genome_q3.csv')

    df_movies_ref.to_csv(path_movies_ref, index=False)
    df_genome_ref.to_csv(path_genome_ref, index=False)


def main():
    prepare_question1()
    prepare_question2()
    prepare_question3()


if __name__ == '__main__':
    main()
