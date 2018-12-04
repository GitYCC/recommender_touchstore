import os

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

WORKSPACE = os.path.dirname(__file__)
DIR_PUBLIC_DATA = os.path.join(WORKSPACE, '..', 'src', 'data')


def _get_old_movies():
    df_movies_public = pd.read_csv(os.path.join(DIR_PUBLIC_DATA, 'movies_pub.csv'))
    return df_movies_public.movieId


def _get_new_movies():
    df_movies_private = pd.read_csv(os.path.join(WORKSPACE, 'movies_prv.csv'))
    return df_movies_private.movieId


def prepare_question1():
    """Prepare Question 1.

    Design a system to predict an unknown rating when given `userId` and `movieId`.

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
    df_q1.loc[:, 'rating'].to_csv(path_answer, index=None)


def prepare_question2():
    """Prepare Question 2.

    Design a system to recommend a top-10 `movieId` list which `userId` did not see before.

    """
    top_movies = 10

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
    df = df_ratings_private[selected]
    df['info'] = df.loc[:, ('rating', 'movieId')].values.tolist()
    dict_q2 = (
        df
        .groupby('userId')['info']
        .apply(list)
        .apply(lambda x: sorted(x, reverse=True)[:top_movies] if len(x) >= top_movies else np.nan)
        .dropna()
        .to_dict()
    )

    path_test = os.path.join(DIR_PUBLIC_DATA, 'test_q2.txt')
    path_answer = os.path.join(WORKSPACE, 'ans_q2.txt')

    fw_test = open(path_test, 'w')
    fw_answer = open(path_answer, 'w')

    for u, list_ in dict_q2.items():
        fw_test.write('{}\n'.format(int(u)))
        movies = list(map(lambda x: str(int(x[1])), list_))
        fw_answer.write('{}\n'.format(','.join(movies)))

    fw_test.close()
    fw_answer.close()


def prepare_question3():
    """Prepare Question 3.

    Design a system to recommend a top-10 `userId` list to a new movie
    when given some information of that new movie.

    """
    top_users = 10

    df_ratings_public = pd.read_csv(os.path.join(DIR_PUBLIC_DATA, 'ratings_pub.csv'))
    df_ratings_private = pd.read_csv(os.path.join(WORKSPACE, 'ratings_prv.csv'))

    old_users = df_ratings_public['userId'].unique()
    new_movies = _get_new_movies()
    common = df_ratings_private.loc[:, ('userId', 'movieId')].merge(
        df_ratings_public.loc[:, ('userId', 'movieId')], on=('userId', 'movieId'))

    selected = (
        df_ratings_private.userId.isin(old_users) &
        df_ratings_private.movieId.isin(new_movies) &
        ((~df_ratings_private.userId.isin(common.userId)) &
         (~df_ratings_private.movieId.isin(common.movieId)))
    )
    df = df_ratings_private[selected]
    df['info'] = df.loc[:, ('rating', 'userId')].values.tolist()
    dict_q3 = (
        df
        .groupby('movieId')['info']
        .apply(list)
        .apply(lambda x: sorted(x, reverse=True)[:top_users] if len(x) >= top_users else np.nan)
        .dropna()
        .to_dict()
    )

    path_test = os.path.join(DIR_PUBLIC_DATA, 'test_q3.txt')
    path_answer = os.path.join(WORKSPACE, 'ans_q3.txt')

    fw_test = open(path_test, 'w')
    fw_answer = open(path_answer, 'w')

    for u, list_ in dict_q3.items():
        fw_test.write('{}\n'.format(int(u)))
        movies = list(map(lambda x: str(int(x[1])), list_))
        fw_answer.write('{}\n'.format(','.join(movies)))

    fw_test.close()
    fw_answer.close()

    selected_movies = np.array(list(map(int, dict_q3.keys())))

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
