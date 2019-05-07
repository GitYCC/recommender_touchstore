import logging
from tempfile import TemporaryDirectory
from collections import defaultdict

import numpy as np
import pandas as pd

import tracer
import process
from process import Datagroup
import decorators
import models
from evaluate import RatingEvaluator, RecommendEvaluator

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_run_id_of_datagroup(train_start_year, valid_start_year):
    run_id = tracer.get_run_id_from_param('datagroup',
                                          {'train_start_year': train_start_year,
                                           'valid_start_year': valid_start_year})
    return run_id


def _get_public_datagroup():
    datagroup = Datagroup(ratings=process.get_ratings(),
                          likes=process.get_likes(),
                          movie_feature=process.get_movie_feature(),
                          user_feature=None)
    return datagroup


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
    logger.info('[prepare datagroup] train from {} to {}, valid from {}'
                .format(train_start_year, valid_start_year, valid_start_year))

    run_id = _get_run_id_of_datagroup(train_start_year, valid_start_year)
    if run_id is not None:
        return run_id

    tracer.start_trace('datagroup')
    tracer.log_param('train_start_year', train_start_year)
    tracer.log_param('valid_start_year', valid_start_year)

    datagroup = _get_public_datagroup()
    _, datagroup_after = process.split_datagroup(train_start_year, datagroup)
    train_group, valid_group = process.split_datagroup(valid_start_year, datagroup_after)

    with TemporaryDirectory() as temp_dir:
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


def _prepare_recommend_problem(um_pair):
    """Convert rating problem to recommend problem."""
    users = np.unique(um_pair[:, 0]).tolist()
    movies = np.unique(um_pair[:, 1]).tolist()
    actions = um_pair.tolist()
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


def _get_fitting_params(datagroup, problem_type):
    if problem_type == 'rating_problem':
        df = datagroup.ratings
        kpi = 'rating'
    elif problem_type == 'like_problem':
        df = datagroup.likes
        kpi = 'like'

    user_movie_pair = df[['userId', 'movieId']].values
    y = df[kpi].values

    movie_feature = datagroup.movie_feature
    user_feature = datagroup.user_feature
    return (user_movie_pair, y, user_feature, movie_feature)


def train(datagroup_id, model_method, topic,
          model_params=None, decorate_method=None, decorate_params=None):
    """Train model.

    Args:
        datagroup_id (int): Datagroup ID.
        model_method (str): Model method defined in models/.
        topic (str): "question1", "question2" or "question3"
        model_params (dict, optional): Parameters required in model training.
        decorate_method (str, optional): Decorating method defined in decorators.py.
        decorate_params (dict, optional): Parameters required in data decorating.

    Returns: None

    """
    if not model_params:
        model_params = dict()
    if not decorate_params:
        decorate_params = dict()

    logger.info('[train model] datagroup_id={}, model_method={}, topic={}, model_params={}, '
                'decorate_method={}, decorate_params={}'
                .format(datagroup_id, model_method, topic, model_params,
                        decorate_method, decorate_params))

    # prepare data
    logger.info('prepare data')

    path_datagroup = tracer.get_artifact_path(datagroup_id, '.')
    train_datagroup = process.load_datagroup(path_datagroup, 'train')
    valid_datagroup = process.load_datagroup(path_datagroup, 'valid')
    evaluate_datagroup = process.load_datagroup(path_datagroup, 'valid')

    if topic == 'question1':
        problem_type = 'rating_problem'
    else:
        problem_type = 'like_problem'

    # data decorate
    if decorate_method:
        decorator_class = getattr(decorators, decorate_method)
        decorator = decorator_class()

        train_datagroup = decorator.decorate(
            train_datagroup, problem_type=problem_type, **decorate_params)
        valid_datagroup = decorator.decorate(
            valid_datagroup, problem_type=problem_type, **decorate_params)

    # fitting
    logger.info('fit model')

    um_pair_train, y_train, u_feature_train, m_feature_train = \
        _get_fitting_params(train_datagroup, problem_type)
    um_pair_valid, y_valid, u_feature_valid, m_feature_valid = \
        _get_fitting_params(valid_datagroup, problem_type)

    model_class = getattr(models, model_method)
    model = model_class()

    model.fit(um_pair_train, y_train, u_feature_train, m_feature_train,
              um_pair_valid, y_valid, u_feature_valid, m_feature_valid,
              **model_params)

    # evaluation
    logger.info('evaluation model')
    um_pair_eval, y_eval, u_feature_eval, m_feature_eval = \
        _get_fitting_params(evaluate_datagroup, problem_type)

    if topic == 'question1':
        valid_result = \
            _evaluate_question1(model, um_pair_eval, y_eval, u_feature_eval, m_feature_eval)
    elif topic == 'question2':
        users_eval, movies_eval, actions_eval = \
            _prepare_recommend_problem(um_pair_eval)
        valid_result = _evaluate_question2(model, users_eval, movies_eval, actions_eval,
                                           u_feature_eval, m_feature_eval)
    elif topic == 'question3':
        users_eval, movies_eval, actions_eval = \
            _prepare_recommend_problem(um_pair_eval)
        valid_result = _evaluate_question3(model, users_eval, movies_eval, actions_eval,
                                           u_feature_eval, m_feature_eval)

    # logging
    logger.info('logging: valid_result={}'.format(valid_result))

    tracer.start_trace('train')
    tracer.log_param('datagroup_id', datagroup_id)
    tracer.log_param('model_method', model_method)
    tracer.log_param('topic', topic)
    tracer.log_param('decorate_method', decorate_method)
    for key, val in model_params.items():
        tracer.log_param(key, val)

    for key, val in valid_result.items():
        tracer.log_metric('valid.{}'.format(key), val)

    tracer.end_trace()


def deploy(model_method, topic,
           model_params=None, decorate_method=None, decorate_params=None):
    """Deploy model.

    Args:
        model_method (str): Model method defined in models/.
        topic (str): "question1", "question2" or "question3"
        model_params (dict, optional): Parameters required in model training.
        decorate_method (str, optional): Decorating method defined in decorators.py.
        decorate_params (dict, optional): Parameters required in data decorating.

    Returns:
        run_id (str): Run ID of deploy.

    """
    if not model_params:
        model_params = dict()
    if not decorate_params:
        decorate_params = dict()

    logger.info(
        '[deploy model] model_method={}, topic={},'
        ' model_params={}, decorate_method={}, decorate_params={}'
        .format(model_method, topic, model_params, decorate_method, decorate_params))

    tracer.start_trace('deploy')
    tracer.log_param('model_method', model_method)
    tracer.log_param('topic', topic)
    tracer.log_param('decorate_method', decorate_method)
    for key, val in model_params.items():
        tracer.log_param(key, val)

    # prepare data
    logger.info('prepare data')

    datagroup = _get_public_datagroup()

    if topic == 'question1':
        problem_type = 'rating_problem'
    else:
        problem_type = 'like_problem'

    # data decorate
    if decorate_method:
        decorator_class = getattr(decorators, decorate_method)
        decorator = decorator_class()
        datagroup = decorator.decorate(datagroup, problem_type=problem_type, **decorate_params)

    # fitting
    logger.info('fit model')

    um_pair, y, u_feature, m_feature = _get_fitting_params(datagroup, problem_type)

    model_class = getattr(models, model_method)
    model = model_class()

    model.fit(um_pair, y, u_feature, m_feature, **model_params)

    # save
    run_id = tracer.get_current_run_id()
    logger.info('save model: run_id={}'.format(run_id))

    tracer.log_model(model)

    tracer.end_trace()
    return run_id


def test(deploy_id):
    """Test model.

    Args:
        deploy_id (str): Run ID of deploy.

    Returns: None

    """
    logger.info('[test depolyed model] deploy_id={}'.format(deploy_id))

    tracer.start_trace('test')
    tracer.log_param('deploy_id', deploy_id)

    deploy_params = tracer.load_params(deploy_id)

    for key, val in deploy_params.items():
        tracer.log_param(key, val)

    model_method = deploy_params['model_method']
    topic = deploy_params['topic']

    # load model
    logger.info('load model')

    model_class = getattr(models, model_method)
    model = tracer.load_model(deploy_id, model_class)

    # evaluation
    logger.info('evaluation model')

    if topic == 'question1':
        datagroup = _get_public_datagroup()

        _, _, u_feature, m_feature = _get_fitting_params(datagroup, 'rating_problem')

        df = process.get_question1()
        um_pair = df[['userId', 'movieId']].values

        y = process.get_answer1()

        result = _evaluate_question1(model, um_pair, y, u_feature, m_feature)

    elif topic == 'question2':
        datagroup = _get_public_datagroup()

        um_pair, _, u_feature, m_feature = _get_fitting_params(datagroup, 'like_problem')

        movies = np.unique(um_pair[:, 1]).tolist()

        df = process.get_question2()
        users = df['userId'].tolist()

        actions = []
        for i, movies in enumerate(process.get_answer2()):
            for movie in movies:
                actions.append((users[i], movie))

        result = _evaluate_question2(model, users, movies, actions, u_feature, m_feature)

    elif topic == 'question3':
        datagroup_old = _get_public_datagroup()
        df_ref_movie_feature = process.get_question3_ref()
        datagroup_new = Datagroup(ratings=pd.DataFrame(
                                    {'userId': [], 'movieId': [],
                                     'rating': [], 'timestamp': []}
                                  ),
                                  likes=pd.DataFrame(
                                    {'userId': [], 'movieId': [],
                                     'like': [], 'timestamp': []}
                                  ),
                                  movie_feature=df_ref_movie_feature,
                                  user_feature=None)

        um_pair, _, u_feature, _ = _get_fitting_params(datagroup_old, 'like_problem')
        _, _, _, m_feature = _get_fitting_params(datagroup_new, 'like_problem')

        users = np.unique(um_pair[:, 0]).tolist()

        df = process.get_question3()
        movies = df['movieId'].tolist()

        actions = []
        for i, users in enumerate(process.get_answer3()):
            for user in users:
                actions.append((user, movies[i]))

        result = _evaluate_question3(model, users, movies, actions, u_feature, m_feature)

    # logging
    logger.info('logging: result={}'.format(result))

    for key, val in result.items():
        tracer.log_metric('test.{}'.format(key), val)

    tracer.end_trace()
