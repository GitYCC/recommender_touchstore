from tempfile import TemporaryDirectory
from collections import defaultdict

import numpy as np
import pandas as pd

import tracer
import process
from process import Datagroup
import converters
import models
from evaluate import RatingEvaluator, RecommendEvaluator


def _get_run_id_of_datagroup(train_start_year, valid_start_year):
    run_id = tracer.get_run_id_from_param('datagroup',
                                          {'train_start_year': train_start_year,
                                           'valid_start_year': valid_start_year})
    return run_id


def prepare_datagroup(train_start_year, valid_start_year):
    """Prepare datagroup.

    Args:
        train_start_year (int): A year of training start time.
        valid_start_year (int): A year of validation start time.

    Note:
        Range of training dataset: train_start_year ~ valid_start_year
        Range of validation dataset: valid_start_year ~ end time of datagroup

    Returns: None

    """
    run_id = _get_run_id_of_datagroup(train_start_year, valid_start_year)
    if run_id is not None:
        return run_id

    tracer.start_trace('datagroup')
    tracer.log_param('train_start_year', train_start_year)
    tracer.log_param('valid_start_year', valid_start_year)

    datagroup = Datagroup(ratings=process.get_ratings(),
                          tags=process.get_tags(),
                          movies=process.get_movies(),
                          genome=process.get_genome())
    _, datagroup_after = process.split_datagroup(train_start_year, datagroup)
    train_group, valid_group = process.split_datagroup(valid_start_year, datagroup_after)

    with TemporaryDirectory(dir='tmp') as temp_dir:
        paths = process.save_datagroup(temp_dir, train_group, 'train')
        for path in paths:
            tracer.log_artifact(path)
        paths = process.save_datagroup(temp_dir, valid_group, 'valid')
        for path in paths:
            tracer.log_artifact(path)

    run_id = tracer.get_current_run_id()
    tracer.end_trace()

    return run_id


def _evaluate_question1(model, um_pair, y, u_feature, m_feature):
    result = dict()
    pred_y = model.predict(um_pair, u_feature, m_feature)
    evaluator = RatingEvaluator(um_pair, y, pred_y)
    result['rms'] = evaluator.get_rms()
    return result


def _prepare_recommend_problem(um_pair, y):
    """Convert rating problem to recommend problem."""
    like_theshold = 3.0
    users = np.unique(um_pair[:, 0]).tolist()
    movies = np.unique(um_pair[:, 1]).tolist()
    actions = (um_pair[y >= like_theshold]).tolist()
    return (users, movies, actions)


def _evaluate_question2(model, users, movies, actions, u_feature, m_feature):
    evaluation_size = 10

    rec_items, rec_scores = \
        model.recommend('movie', users, movies, u_feature, m_feature, maxsize=evaluation_size)

    user_action_dict = defaultdict(lambda: list())
    for userId, movieId in actions:
        user_action_dict[userId].append(movieId)
    user_actions = [user_action_dict[userId] for userId in users]

    evaluator = RecommendEvaluator(users, user_actions, rec_items, rec_scores)

    result = dict()
    result['map_t{}'.format(evaluation_size)] = \
        evaluator.get_mean_average_precision(size=evaluation_size)

    return result


def _evaluate_question3(model, users, movies, actions, u_feature, m_feature):
    evaluation_size = 10

    rec_items, rec_scores = \
        model.recommend('user', users, movies, u_feature, m_feature, maxsize=evaluation_size)

    movie_action_dict = defaultdict(lambda: list())
    for userId, movieId in actions:
        movie_action_dict[movieId].append(userId)
    movie_actions = [movie_action_dict[movieId] for movieId in movies]

    evaluator = RecommendEvaluator(movies, movie_actions, rec_items, rec_scores)

    result = dict()
    result['map_t{}'.format(evaluation_size)] = \
        evaluator.get_mean_average_precision(size=evaluation_size)

    return result


def train(datagroup_id, convert_method, model_method, topic, model_params=None):
    """Train model.

    Args:
        datagroup_id (int): Datagroup ID.
        convert_method (str): Convert method defined in converters.py.
        model_method (str): Model method defined in models/.
        topic (str): "question1", "question2" or "question3"
        model_params (dict, optional): Parameters required in model training.

    Returns: None

    """
    if model_params is None:
        model_params = dict()

    tracer.start_trace('train')
    tracer.log_param('datagroup_id', datagroup_id)
    tracer.log_param('convert_method', convert_method)
    tracer.log_param('model_method', model_method)
    tracer.log_param('topic', topic)
    for key, val in model_params.items():
        tracer.log_param(key, val)

    # prepare data
    path_datagroup = tracer.get_artifact_path(datagroup_id, '.')
    train_datagroup = process.load_datagroup(path_datagroup, 'train')
    valid_datagroup = process.load_datagroup(path_datagroup, 'valid')
    conv_class = getattr(converters, convert_method)
    conv = conv_class()

    um_pair_train, y_train, u_feature_train, m_feature_train = conv.convert(train_datagroup)
    um_pair_valid, y_valid, u_feature_valid, m_feature_valid = conv.convert(valid_datagroup)

    # fitting
    model_class = getattr(models, model_method)
    model = model_class(**model_params)

    model.fit(um_pair_train, y_train, u_feature_train, m_feature_train)

    # evaluation
    if topic == 'question1':
        train_result = \
            _evaluate_question1(model, um_pair_train, y_train, u_feature_train, m_feature_train)
        valid_result = \
            _evaluate_question1(model, um_pair_valid, y_valid, u_feature_valid, m_feature_valid)
    elif topic == 'question2':
        users_train, movies_train, actions_train = \
            _prepare_recommend_problem(um_pair_train, y_train)
        train_result = _evaluate_question2(model, users_train, movies_train, actions_train,
                                           u_feature_train, m_feature_train)
        users_valid, movies_valid, actions_valid = \
            _prepare_recommend_problem(um_pair_valid, y_valid)
        valid_result = _evaluate_question2(model, users_valid, movies_valid, actions_valid,
                                           u_feature_valid, m_feature_valid)
    elif topic == 'question3':
        users_train, movies_train, actions_train = \
            _prepare_recommend_problem(um_pair_train, y_train)
        train_result = _evaluate_question3(model, users_train, movies_train, actions_train,
                                           u_feature_train, m_feature_train)
        users_valid, movies_valid, actions_valid = \
            _prepare_recommend_problem(um_pair_valid, y_valid)
        valid_result = _evaluate_question3(model, users_valid, movies_valid, actions_valid,
                                           u_feature_valid, m_feature_valid)

    # logging
    for key, val in train_result.items():
        tracer.log_metric('train.{}'.format(key), val)

    for key, val in valid_result.items():
        tracer.log_metric('valid.{}'.format(key), val)

    tracer.end_trace()


def deploy(convert_method, model_method, topic, model_params=None):
    """Deploy model.

    Args:
        convert_method (str): Convert method defined in converters.py.
        model_method (str): Model method defined in models/.
        topic (str): "question1", "question2" or "question3"
        model_params (dict, optional): Parameters required in model training.

    Returns: None

    """
    if model_params is None:
        model_params = dict()

    tracer.start_trace('deploy')
    tracer.log_param('convert_method', convert_method)
    tracer.log_param('model_method', model_method)
    tracer.log_param('topic', topic)
    for key, val in model_params.items():
        tracer.log_param(key, val)

    # prepare data
    datagroup = Datagroup(ratings=process.get_ratings(),
                          tags=process.get_tags(),
                          movies=process.get_movies(),
                          genome=process.get_genome())

    conv_class = getattr(converters, convert_method)
    conv = conv_class()
    um_pair, y, u_feature, m_feature = conv.convert(datagroup)

    # fitting
    model_class = getattr(models, model_method)
    model = model_class(**model_params)

    model.fit(um_pair, y, u_feature, m_feature)

    # save
    tracer.log_model(model)

    tracer.end_trace()


def test(deploy_id):
    """Test model.

    Args:
        deploy_id (str): Run ID of deploy.

    Returns: None

    """
    tracer.start_trace('test')
    tracer.log_param('deploy_id', deploy_id)

    deploy_params = tracer.load_params(deploy_id)
    print(deploy_params)
    model_method = deploy_params['model_method']
    topic = deploy_params['topic']
    convert_method = deploy_params['convert_method']

    tracer.log_param('topic', topic)
    tracer.log_param('convert_method', convert_method)

    # load model
    model_class = getattr(models, model_method)
    model = tracer.load_model(deploy_id, model_class)

    # evaluation
    if topic == 'question1':
        datagroup = Datagroup(ratings=process.get_ratings(),
                              tags=process.get_tags(),
                              movies=process.get_movies(),
                              genome=process.get_genome())
        conv_class = getattr(converters, convert_method)
        conv = conv_class()
        _, _, u_feature, m_feature = conv.convert(datagroup)

        df = process.get_question1()
        um_pair = df[['userId', 'movieId']].values

        y = process.get_answer1()

        result = _evaluate_question1(model, um_pair, y, u_feature, m_feature)

    elif topic == 'question2':
        datagroup = Datagroup(ratings=process.get_ratings(),
                              tags=process.get_tags(),
                              movies=process.get_movies(),
                              genome=process.get_genome())
        conv_class = getattr(converters, convert_method)
        conv = conv_class()
        um_pair, _, u_feature, m_feature = conv.convert(datagroup)

        movies = np.unique(um_pair[:, 1]).tolist()

        df = process.get_question2()
        users = df['userId'].tolist()

        actions = []
        for i, movies in enumerate(process.get_answer2()):
            for movie in movies:
                actions.append((users[i], movie))

        result = _evaluate_question2(model, users, movies, actions, u_feature, m_feature)

    elif topic == 'question3':
        datagroup_old = Datagroup(ratings=process.get_ratings(),
                                  tags=process.get_tags(),
                                  movies=process.get_movies(),
                                  genome=process.get_genome())
        df_ref_movies, df_ref_genome = process.get_question3_ref()
        datagroup_new = Datagroup(ratings=pd.Dateframe(
                                    {'userId': [], 'movieId': [],
                                     'rating': [], 'timestamp': []}
                                  ),
                                  tags=process.get_tags(
                                    {'userId': [], 'movieId': [],
                                     'tag': [], 'timestamp': []}
                                  ),
                                  movies=df_ref_movies,
                                  genome=df_ref_genome)

        conv_class = getattr(converters, convert_method)
        conv = conv_class()
        um_pair, _, u_feature, _ = conv.convert(datagroup_old)
        _, _, _, m_feature = conv.convert(datagroup_new)

        users = np.unique(um_pair[:, 0]).tolist()

        df = process.get_question3()
        movies = df['movieId'].tolist()

        actions = []
        for i, users in enumerate(process.get_answer3()):
            for user in users:
                actions.append((user, movies[i]))

        result = _evaluate_question2(model, users, movies, actions, u_feature, m_feature)

    for key, val in result.items():
        tracer.log_metric('test.{}'.format(key), val)

    tracer.end_trace()
