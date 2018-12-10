import os
from datetime import datetime
from collections import namedtuple
import tempfile

import tracer
import process

Datagroup = namedtuple('Datagroup', ['ratings', 'tags', 'movies', 'genome'])


def _get_run_id_of_dataset(train_start_year, valid_start_year):
    run_id = tracer.get_run_id_from_param('dataset',
                                          {'train_start_year': train_start_year,
                                           'valid_start_year': valid_start_year})
    return run_id


def _split_datagroup(cut_year, datagroup):
    cut_dt = datetime(cut_year, 1, 1)
    df_ratings = datagroup.ratings
    df_tags = datagroup.tags
    df_movies = datagroup.movies
    df_genome = datagroup.genome

    df_ratings_before, df_ratings_after = \
        process.split_data_by_datetime(df_ratings, cut_dt)
    df_tags_before, df_tags_after = \
        process.split_data_by_datetime(df_tags, cut_dt)
    df_movies_before, df_movies_after = \
        process.split_data_by_year(df_movies, cut_year)
    df_genome_before = process.select_data_by_movie_group(df_genome,
                                                          df_movies_before.movieId)
    df_genome_after = process.select_data_by_movie_group(df_genome,
                                                         df_movies_after.movieId)

    datagroup_before = Datagroup(ratings=df_ratings_before,
                                 tags=df_tags_before,
                                 movies=df_movies_before,
                                 genome=df_genome_before)
    datagroup_after = Datagroup(ratings=df_ratings_after,
                                tags=df_tags_after,
                                movies=df_movies_after,
                                genome=df_genome_after)
    return (datagroup_before, datagroup_after)


def save_datagroup(datagroup, postfix):
    temp_dir = tempfile.mkdtemp()
    paths = list()
    for key in ['ratings', 'tags', 'movies', 'genome']:
        df = getattr(datagroup, key)
        path = os.path.join(temp_dir, key + '_' + postfix + '.csv')
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def prepare_dataset(train_start_year, valid_start_year):
    run_id = _get_run_id_of_dataset(train_start_year, valid_start_year)
    if run_id is not None:
        return run_id

    tracer.start_trace('dataset')
    tracer.log_param('train_start_year', train_start_year)
    tracer.log_param('valid_start_year', valid_start_year)

    datagroup = Datagroup(ratings=process.get_ratings(),
                          tags=process.get_tags(),
                          movies=process.get_movies(),
                          genome=process.get_genome())
    _, datagroup_after = _split_datagroup(train_start_year, datagroup)
    train_group, valid_group = _split_datagroup(valid_start_year, datagroup_after)

    paths = save_datagroup(train_group, 'train')
    for path in paths:
        tracer.log_artifact(path)
    paths = save_datagroup(valid_group, 'valid')
    for path in paths:
        tracer.log_artifact(path)

    run_id = tracer.get_current_run_id()
    tracer.end_trace()

    return run_id
