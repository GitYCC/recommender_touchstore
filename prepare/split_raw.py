import os
from datetime import datetime

import pandas as pd

WORKSPACE = os.path.dirname(__file__)
DIR_MovieLens20M = os.path.join(WORKSPACE, 'ml-20m')
CUT_DATETIME = datetime(2013, 1, 1)
CUT_YEAR = 2013
CUT_TIMESTAMP = int((CUT_DATETIME - datetime(1970, 1, 1)).total_seconds())
DIR_PUBLIC_DATA = os.path.join(WORKSPACE, '..', 'src', 'data')


def _split_by_timestamp(df, cut_timestamp):
    df_public = df[df.timestamp < cut_timestamp]
    df_private = df[df.timestamp >= cut_timestamp]
    return (df_public, df_private)


def _select_by_movie_group(df, movie_group):
    return df[df.movieId.isin(movie_group)]


def main():
    # load data
    df_ratings = pd.read_csv(os.path.join(DIR_MovieLens20M, 'ratings.csv'))
    df_movies = pd.read_csv(os.path.join(DIR_MovieLens20M, 'movies.csv'))
    df_tags = pd.read_csv(os.path.join(DIR_MovieLens20M, 'tags.csv'))
    df_links = pd.read_csv(os.path.join(DIR_MovieLens20M, 'links.csv'))
    df_genome_scores = pd.read_csv(os.path.join(DIR_MovieLens20M, 'genome-scores.csv'))
    df_genome_tags = pd.read_csv(os.path.join(DIR_MovieLens20M, 'genome-tags.csv'))

    # process
    df_ratings_public, df_ratings_private = _split_by_timestamp(df_ratings, CUT_TIMESTAMP)

    df_tags_public, df_tags_private = _split_by_timestamp(df_tags, CUT_TIMESTAMP)

    year_series = (df_movies['title'].str.extract(r'\((....)\) *$'))[0].astype('float32')
    old_movies = df_movies[year_series < CUT_YEAR]['movieId'].unique()
    new_movies = df_movies[year_series >= CUT_YEAR]['movieId'].unique()

    df_movies_public = _select_by_movie_group(df_movies, old_movies)
    df_movies_private = _select_by_movie_group(df_movies, new_movies)

    df_links_public = _select_by_movie_group(df_links, old_movies)
    df_links_private = _select_by_movie_group(df_links, new_movies)

    df_genome = df_genome_scores.merge(df_genome_tags, how='left')
    df_genome_public = _select_by_movie_group(df_genome, old_movies)
    df_genome_private = _select_by_movie_group(df_genome, new_movies)

    # save public data
    df_ratings_public.to_csv(os.path.join(DIR_PUBLIC_DATA, 'ratings_pub.csv'), index=None)
    df_tags_public.to_csv(os.path.join(DIR_PUBLIC_DATA, 'tags_pub.csv'), index=None)
    df_movies_public.to_csv(os.path.join(DIR_PUBLIC_DATA, 'movies_pub.csv'), index=None)
    df_links_public.to_csv(os.path.join(DIR_PUBLIC_DATA, 'links_pub.csv'), index=None)
    df_genome_public.to_csv(os.path.join(DIR_PUBLIC_DATA, 'genome_pub.csv'), index=None)

    # save private data
    df_ratings_private.to_csv(os.path.join(WORKSPACE, 'ratings_prv.csv'), index=None)
    df_tags_private.to_csv(os.path.join(WORKSPACE, 'tags_prv.csv'), index=None)
    df_movies_private.to_csv(os.path.join(WORKSPACE, 'movies_prv.csv'), index=None)
    df_links_private.to_csv(os.path.join(WORKSPACE, 'links_prv.csv'), index=None)
    df_genome_private.to_csv(os.path.join(WORKSPACE, 'genome_prv.csv'), index=None)


if __name__ == '__main__':
    main()
