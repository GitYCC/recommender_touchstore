import os
from tempfile import TemporaryDirectory
from pathlib import Path

import yaml
import mlflow
from mlflow.tracking.fluent import active_run


IS_LOGABLE = True


class _Controller:
    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        if IS_LOGABLE:
            return self._func(*args, **kwargs)
        else:
            return


class _MLFlowPath:
    def __init__(self):
        self.path_mlrun = self._get_path_mlrun()
        self.exp_info = self._get_exp_info(self.path_mlrun)
        self.run_info = self._get_run_info(self.exp_info)

    def _get_path_mlrun(self):
        path_mlrun = mlflow.tracking.get_tracking_uri()
        if os.path.exists(path_mlrun):
            return Path(path_mlrun)
        else:
            return None

    def _get_exp_info(self, path_mlrun):
        if path_mlrun is None:
            return None

        def is_int(x):
            try:
                x = int(x)
            except ValueError:
                return False
            return True
        paths = [child for child in path_mlrun.iterdir() if is_int(child.name)]

        exp_info = dict()
        for exp_path in paths:
            info = dict()
            info['path'] = exp_path
            info['runs'] = [child.name for child in exp_path.iterdir()
                            if child.is_dir() and len(child.name) == 32]

            path_meta = exp_path / 'meta.yaml'
            name = None
            with path_meta.open('r') as fr:
                name = yaml.load(fr)['name']
            exp_info[name] = info
        return exp_info

    def _get_run_info(self, exp_info):
        if exp_info is None:
            return None

        run_info = dict()
        for exp_name, exp_info in exp_info.items():
            for run_id in exp_info['runs']:
                info = dict()

                info['parent_exp'] = exp_name

                run_path = exp_info['path'] / run_id
                info['path'] = run_path

                params_paths = [child for child in (run_path / 'params').iterdir()]
                params_list = [(p.name, p.open('r').readline().strip()) for p in params_paths]
                info['params'] = sorted(params_list)

                run_info[run_id] = info
        return run_info

    def get_path_of_exp(self, exp_name):
        if self.exp_info is None:
            return None
        return self.exp_info[exp_name]['path']

    def get_path_of_run(self, run_id):
        if self.run_info is None:
            return None
        return self.run_info[run_id]['path']

    def get_run_id_from_param(self, exp_name, param_dict):
        if self.run_info is None:
            return None

        param_list = sorted([(name, str(val)) for name, val in param_dict.items()])
        for run_id, info in self.run_info.items():
            if info['parent_exp'] == exp_name and info['params'] == param_list:
                return run_id
        return None


@_Controller
def start_trace(job_name=None):
    if job_name is not None:
        mlflow.set_experiment(job_name)
        mlflow.start_run()
    else:
        mlflow.start_run()


@_Controller
def end_trace():
    mlflow.end_run()


@_Controller
def get_current_run_id():
    return active_run().info.run_uuid


def get_run_id_from_param(job_name, param_dict):
    return _MLFlowPath().get_run_id_from_param(job_name, param_dict)


@_Controller
def log_param(key, val):
    mlflow.log_param(key, val)


def load_params(run_id):
    param_list = _MLFlowPath().run_info[run_id]['params']
    param_dict = {key: val for key, val in param_list}
    return param_dict


@_Controller
def log_metric(key, val):
    mlflow.log_metric(key, val)


@_Controller
def log_artifact(fname):
    mlflow.log_artifact(fname)


def get_artifact_path(run_id, fname):
    return _MLFlowPath().get_path_of_run(run_id) / 'artifacts' / fname


@_Controller
def log_model(model):
    with TemporaryDirectory(dir='tmp') as tmp_dir:
        model.save(Path(tmp_dir))
        mlflow.log_artifacts(tmp_dir, artifact_path='model')


def load_model(run_id, model_class):
    model_dir = _MLFlowPath().get_path_of_run(run_id) / 'artifacts' / 'model'
    return model_class.load(model_dir)
