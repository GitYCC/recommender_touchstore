from tempfile import TemporaryDirectory
import os

import numpy as np

import tracer
import process
from process import Datagroup
import converters
import models
from evaluate import Evaluator
import config


def _get_run_id_of_datagroup(train_start_year, valid_start_year):
    run_id = tracer.get_run_id_from_param('datagroup',
                                          {'train_start_year': train_start_year,
                                           'valid_start_year': valid_start_year})
    return run_id


def prepare_datagroup(train_start_year, valid_start_year):
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


def _convert_datagroup(datagroup_id, method):
    path_datagroup = tracer.load_artifact(datagroup_id, '.')
    train_datagroup = process.load_datagroup(path_datagroup, 'train')
    valid_datagroup = process.load_datagroup(path_datagroup, 'valid')
    conv_class = getattr(converters, method)
    conv = conv_class()
    return (conv.convert(train_datagroup), conv.convert(valid_datagroup))


def train(datagroup_id, convert_method, model_method, evaluate_methods, model_params=None):
    if model_params is None:
        model_params = dict()

    tracer.start_trace('train')
    tracer.log_param('datagroup_id', datagroup_id)
    tracer.log_param('convert_method', convert_method)
    tracer.log_param('model_method', model_method)
    for key, val in model_params.items():
        tracer.log_param(key, val)

    converted_train, converted_valid = _convert_datagroup(datagroup_id, convert_method)
    um_pair_train, y_train, u_feature_train, m_feature_train = converted_train
    um_pair_valid, y_valid, u_feature_valid, m_feature_valid = converted_valid

    model_class = getattr(models, model_method)
    model = model_class(**model_params)

    model.fit(um_pair_train, y_train, u_feature_train, m_feature_train)

    for evaluate_method in evaluate_methods:
        evaluator = Evaluator(model, um_pair_valid, y_valid, u_feature_valid, m_feature_valid)
        result = getattr(evaluator, 'get_{}'.format(evaluate_method))()
        tracer.log_metric('valid.{}'.format(evaluate_method), result)

    tracer.end_trace()


def deploy(convert_method, model_method, evaluate_methods, model_params=None):
    if model_params is None:
        model_params = dict()

    tracer.start_trace('deploy')
    tracer.log_param('convert_method', convert_method)
    tracer.log_param('model_method', model_method)
    for key, val in model_params.items():
        tracer.log_param(key, val)

    datagroup = Datagroup(ratings=process.get_ratings(),
                          tags=process.get_tags(),
                          movies=process.get_movies(),
                          genome=process.get_genome())

    conv_class = getattr(converters, convert_method)
    conv = conv_class()
    um_pair, y, u_feature, m_feature = conv.convert(datagroup)

    model_class = getattr(models, model_method)
    model = model_class(**model_params)

    model.fit(um_pair, y, u_feature, m_feature)
    tracer.log_model(model)

    for evaluate_method in evaluate_methods:
        evaluator = Evaluator(model, um_pair, y, u_feature, m_feature)
        result = getattr(evaluator, 'get_{}'.format(evaluate_method))()
        tracer.log_metric('valid.{}'.format(evaluate_method), result)

    tracer.end_trace()


def test_question1(deploy_id, model_method):
    tracer.start_trace('test_question1')
    tracer.log_param('deploy_id', deploy_id)
    tracer.log_param('model_method', model_method)

    model_class = getattr(models, model_method)
    model = tracer.load_model(deploy_id, model_class)

    df = process.get_question1()
    um_pair = df[['userId', 'movieId']].values

    ans = list()
    with open(os.path.join(config.DIR_PRIVATE, 'ans_q1.txt')) as fr:
        for line in fr.readlines():
            ans.append(float(line.strip()))
    ans = np.array(ans)

    evaluator = Evaluator(model, um_pair, ans)
    result = evaluator.get_rms()

    tracer.log_metric('rms', result)
    tracer.end_trace()
